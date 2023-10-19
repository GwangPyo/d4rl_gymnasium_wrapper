import gymnasium as gym
import gym as classic_gym
import numpy as np
from inspect import getfullargspec

HAS_TIMEOUT_IN_CLASSIC_GYM = classic_gym.__version__ > '0.24'


class SpaceParser(object):

    @staticmethod
    def discrete(space: classic_gym.spaces.Discrete):
        n = space.n
        dtype = space.dtype
        return gym.spaces.Discrete(n, dtype)

    @staticmethod
    def box(space: classic_gym.spaces.Box,
            cast_float32: bool = True):
        low = space.low
        high = space.high
        shape = space.shape
        if not cast_float32:
            dtype = space.dtype
        else:
            dtype = np.float32
        return gym.spaces.Box(low, high, shape=shape, dtype=dtype)

    @staticmethod
    def multi_discrete(space: classic_gym.spaces.MultiDiscrete):
        nvec = space.nvec
        dtype = space.dtype
        return gym.spaces.MultiDiscrete(nvec=nvec, dtype=dtype)

    @staticmethod
    def multi_binary(space: classic_gym.spaces.MultiBinary):
        n = space.n
        dtype = space.dtype
        return gym.spaces.MultiBinary(n=n, dtype=dtype)

    @staticmethod
    def tuple(space: classic_gym.spaces.Tuple):
        for s in space:
            gym.spaces.Tuple([type_parser[type(s)](s)])

    @staticmethod
    def dict(space: classic_gym.spaces.Dict):
        return gym.spaces.Dict({k: type_parser[type(v)](v) for k, v in space.spaces})

    @staticmethod
    def parse(space, cast_float32: bool = True):
        try:
            parser = type_parser[type(space)]
            spec = getfullargspec(parser)
            if cast_float32 in spec.args:
                parser(space, cast_float32)
            else:
                parser(space)
        except KeyError:
            raise NotImplementedError(f"Type {type(space)} is not implemented yet.")


type_parser = {
    classic_gym.spaces.Discrete: SpaceParser.discrete,
    classic_gym.spaces.Box: SpaceParser.box,

    classic_gym.spaces.MultiDiscrete: SpaceParser.multi_discrete,
    classic_gym.spaces.MultiBinary: SpaceParser.multi_binary,
    classic_gym.spaces.Tuple: SpaceParser.tuple,
    classic_gym.spaces.Dict: SpaceParser.dict,
}


class GymnasiumWrapper(gym.Env):
    render_mode: str

    def __init__(self,
                 wrapped_env,
                 *,
                 timeout_key: str = 'TimeLimit.truncated',
                 **kwargs,
                 ):
        self.wrapped = wrapped_env
        self.timeout_key = timeout_key
        try:
            self.observation_space_type = SpaceParser.parse(self.wrapped.observation_space)
            self.action_space_type = SpaceParser.parse(self.wrapped.action_space)

        except KeyError:
            raise NotImplementedError

    @property
    def unwrapped(self):
        return self.wrapped

    def render(self, render_mode: str = 'human'):
        return self.wrapped.render(mode=render_mode)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.wrapped.seed(seed)
        if options is not None:
            raise NotImplementedError("Setting option is not implemented yet.")

        return self.wrapped.reset(), {}

    def step(self, actions):
        if HAS_TIMEOUT_IN_CLASSIC_GYM:
            observation, reward, done, timeout, info = self.wrapped.step(actions)
            return observation, reward, done, timeout, info
        else:
            observation, reward, done, info = self.wrapped.step(actions)
            if self.timeout_key in info.keys():
                timeout = info.pop(self.timeout_key)
            else:
                timeout = False
            return observation, reward, done, timeout, info


class D4RLGymnasiumEnv(GymnasiumWrapper):
    def __init__(self, env_id):
        import d4rl
        super().__init__(classic_gym.make(env_id))

    def get_dataset(self):
        return self.unwrapped.get_dataset()



