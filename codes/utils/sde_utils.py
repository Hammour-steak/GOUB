import math
import torch
import abc
from tqdm import tqdm
import torchvision.utils as tvutils
import os
from scipy import integrate


class SDE(abc.ABC):
    def __init__(self, T, device=None):
        self.T = T
        self.dt = 1 / T
        self.device = device

    @abc.abstractmethod
    def drift(self, x, t):
        pass

    @abc.abstractmethod
    def dispersion(self, x, t):
        pass

    @abc.abstractmethod
    def sde_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def score_fn(self, x, t):
        pass

    ################################################################################

    def forward_step(self, x, t):
        return x + self.drift(x, t) + self.dispersion(x, t)

    def reverse_sde_step_mean(self, x, score, t):  # train process
        return x - self.sde_reverse_drift(x, score, t)

    def forward(self, x0, T=-1):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

        return x

    def reverse_sde(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_sde_step(x, score, t)

        return x

    def reverse_ode(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_ode_step(x, score, t)
        return x


#############################################################################


class GOUB(SDE):
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''

    def __init__(self, lambda_square, T=100, schedule='cosine', eps=0.01, device=None):
        super().__init__(T, device)
        self.lambda_square = lambda_square / 255 if lambda_square >= 1 else lambda_square
        self._initialize(self.lambda_square, T, schedule, eps)

    def _initialize(self, lambda_square, T, schedule, eps=0.01):

        def constant_theta_schedule(timesteps, v=1.):
            """
            constant schedule
            """
            print('constant schedule')
            timesteps = timesteps + 1  # T from 1 to 100
            return torch.ones(timesteps, dtype=torch.float32)

        def linear_theta_schedule(timesteps):
            """
            linear schedule
            """
            print('linear schedule')
            timesteps = timesteps + 1  # T from 1 to 100
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        def cosine_theta_schedule(timesteps, s=0.008):
            """
            cosine schedule
            """
            print('cosine schedule')
            timesteps = timesteps + 2  # for truncating from 1 to -1
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        def get_sigmas(thetas):
            return torch.sqrt(lambda_square ** 2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(lambda_square ** 2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))

        if schedule == 'cosine':
            thetas = cosine_theta_schedule(T)
        elif schedule == 'linear':
            thetas = linear_theta_schedule(T)
        elif schedule == 'constant':
            thetas = constant_theta_schedule(T)
        else:
            print('Not implemented such schedule yet!!!')

        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0]  # for that thetas[0] is not 0
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)

        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)

        self.sigma_bars = sigma_bars.to(self.device)

        self.sigma_t_T = torch.sqrt(
            self.lambda_square ** 2 * (1 - torch.exp(-2 * (self.thetas_cumsum[-1] - self.thetas_cumsum) * self.dt)))

        self.f_sigmas = self.sigma_bars * self.sigma_t_T / self.sigma_bars[-1]

        self.mu = 0.
        self.model = None

    #####################################

    # set mu for different cases
    def set_mu(self, mu):
        self.mu = mu

    # set score model for reverse process
    def set_model(self, model):
        self.model = model

    def scaled_reverse_sde_step_mean(self, x, noise, t):
        tmp = torch.exp((self.thetas_cumsum[t] - self.thetas_cumsum[-1]) * self.dt)  # e^{-\bar\theta_{t:T}}}
        drift_h = - self.sigmas[t] ** 2 * tmp ** 2 / self.sigma_t_T[t] ** 2 * (x - self.mu)
        mask = (t == 100)
        mask_expanded = mask.expand_as(drift_h)
        drift_h[mask_expanded] = 0
        return self.f_sigma(t) * x - (
                    self.f_sigma(t) * self.thetas[t] * (self.mu - x) + self.f_sigma(t) * drift_h + self.sigmas[
                t] ** 2 * noise) * self.dt

    def scaled_reverse_optimum_step(self, xt_1_optimum, t):
        return self.f_sigma(t) * xt_1_optimum

    def reverse_sde_step(self, x, score, t):  # val process
        return x - self.sde_reverse_drift_1(x, score, t) - self.dispersion(x, t)

    def reverse_mean_ode_step(self, x, score, t):  # val process
        return x - self.sde_reverse_drift_1(x, score, t)

    #####################################

    def m(self, t):  # cofficient of x0 in marginal forward process
        return torch.exp(-self.thetas_cumsum[t] * self.dt) * self.sigma_t_T[t] ** 2 / self.sigma_bars[-1] ** 2

    def n(self, t):  # cofficient of xT in marginal forward process
        return ((1 - torch.exp(-self.thetas_cumsum[t] * self.dt)) * self.sigma_t_T[t] ** 2 + torch.exp(
            -2 * (self.thetas_cumsum[-1] - self.thetas_cumsum[t]) * self.dt) * self.sigma_bars[t] ** 2) / \
               self.sigma_bars[-1] ** 2

    def f_m(self, t):  # cofficient of x_{t-1} in forward process
        return self.m(t) / self.m(t - 1)

    def f_n(self, t):  # cofficient of x_T in forward process
        return self.n(t) - self.n(t - 1) * self.m(t) / self.m(t - 1)

    def f_sigma_1(self, t):  # forward sigma with t : t-1 to t
        return torch.sqrt(self.f_sigma(t) ** 2 - self.f_sigma(t - 1) ** 2 * self.f_m(t) ** 2)

    def f_mean_1(self, xt_1, t):  # forward mean with t : t-1 to t
        return self.f_m(t) * xt_1 + self.f_n(t) * self.mu

    def r_sigma_1(self, t):  # reverse sigma with t : t to t-1
        return self.f_sigma_1(t) * self.f_sigma(t - 1) / self.f_sigma(t)

    def r_mean_1(self, xt, x0, t):  # reverse mean with t : t to t-1
        return (self.f_sigma(t - 1) ** 2 * self.f_m(t) * (xt - self.f_n(t) * self.mu) +
                self.f_sigma_1(t) ** 2 * self.f_mean(x0, t - 1)) / self.f_sigma(t) ** 2

    #####################################

    def mu_bar(self, x0, t):
        return self.mu + (x0 - self.mu) * torch.exp(-self.thetas_cumsum[t] * self.dt)

    def f_mean(self, x0, t):  # forward mean with t
        mean = self.m(t) * x0 + self.n(t) * self.mu
        return mean

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def f_sigma(self, t):  # marginal forward sigma with t
        return self.f_sigmas[t]

    def drift(self, x, t):
        if t == 100:
            return (self.thetas[t] * (self.mu - x)) * self.dt
        # add h-transform term
        tmp = torch.exp(2 * (self.thetas_cumsum[t] - self.thetas_cumsum[-1]) * self.dt)  # e^{-2\bar\theta_{t:T}}}
        drift_h = - self.sigmas[t] ** 2 * tmp / self.sigma_t_T[t] ** 2 * (x - self.mu)
        return (self.thetas[t] * (self.mu - x) + drift_h) * self.dt

    def sde_reverse_drift_1(self, x, score, t):
        # add h-transform term
        if t == 100:
            return (self.thetas[t] * (self.mu - x) - self.sigmas[t] ** 2 * score) * self.dt  # drift_h=0
        tmp = torch.exp(2 * (self.thetas_cumsum[t] - self.thetas_cumsum[-1]) * self.dt)  # e^{-2\bar\theta_{t:T}}}
        drift_h = - self.sigmas[t] ** 2 * tmp / self.sigma_t_T[t] ** 2 * (x - self.mu)
        return (self.thetas[t] * (self.mu - x) + drift_h - self.sigmas[t] ** 2 * score) * self.dt

    def sde_reverse_drift(self, x, score, t):
        # add h-transform term
        tmp = torch.exp(2 * (self.thetas_cumsum[t] - self.thetas_cumsum[-1]) * self.dt)  # e^{-2\bar\theta_{t:T}}}
        drift_h = - self.sigmas[t] ** 2 * tmp / self.sigma_t_T[t] ** 2 * (x - self.mu)
        mask = (t == 100)
        mask_expanded = mask.expand_as(drift_h)
        drift_h[mask_expanded] = 0
        return (self.thetas[t] * (self.mu - x) + drift_h - self.sigmas[t] ** 2 * score) * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x) * math.sqrt(self.dt)).to(self.device)

    def get_score_from_noise(self, noise, t):
        score = - noise / self.f_sigma(t)
        return score

    def score_fn(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        return self.model(x, self.mu, t, **kwargs)

    def noise_fn(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        return self.model(x, self.mu, t, **kwargs)

    def scaled_score(self, score, t):
        return self.sigmas[t] * score

    def get_real_score(self, xt, x0, t):
        real_score = -(xt - self.f_mean(x0, t)) / self.f_sigma(t) ** 2
        return real_score

    # optimum x_{t-1}
    def reverse_optimum_step(self, xt, x0, t):
        mean = self.r_mean_1(xt, x0, t)
        return mean

    def sigma(self, t):
        return self.sigmas[t]

    def theta(self, t):
        return self.thetas[t]

    def get_real_noise(self, xt, x0, t):
        real_noise = (xt - self.f_mean(x0, t)) / self.f_sigma(t)
        mask = (t == 100)
        mask_expanded = mask.expand_as(real_noise)
        real_noise[mask_expanded] = 0
        return real_noise

    # forward process to get x(T) from x(0)
    def forward(self, x0, T=-1, save_dir='forward_state'):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{t}.png', normalize=False)
        return x

    def reverse_sde(self, xt, T=-1, save_states=False, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            noise = self.model(x, self.mu, t, **kwargs)
            score = - noise / self.f_sigma(t) if t != 100 else 0
            x = self.reverse_sde_step(x, score, t)

            if save_states:  # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    def reverse_mean_ode(self, xt, T=-1, save_states=False, save_dir='ode_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            noise = self.model(x, self.mu, t, **kwargs)
            score = - noise / self.f_sigma(t) if t != 100 else 0
            x = self.reverse_mean_ode_step(x, score, t)

            if save_states:  # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x


    ################################################################

    def weights(self, t):
        return torch.exp(-self.thetas_cumsum[t] * self.dt)

    # sample states for training
    def generate_random_states(self, x0, mu):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)
        
        self.set_mu(mu)
        batch = x0.shape[0]
        timesteps = torch.randint(1, self.T, (batch, 1, 1, 1)).long()

        state_mean = self.f_mean(x0, timesteps)
        noises = torch.randn_like(state_mean)
        noise_level = self.f_sigma(timesteps)
        noisy_states = noises * noise_level + state_mean
        return timesteps, noisy_states.to(torch.float32)
