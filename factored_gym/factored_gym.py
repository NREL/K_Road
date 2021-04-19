from typing import (
    Any,
    Optional,
    Tuple,
    )

import gym

from factored_gym.action_transform.action_transform import ActionTransform
from factored_gym.factored_gym_component import FactoredGymComponent
from factored_gym.observer import Observer
from factored_gym.process import Process
from factored_gym.rewarder import Rewarder
from factored_gym.terminator import Terminator
from factored_gym.view import View


class FactoredGym(gym.Env):
    """
    An OpenAI Gym compatible class that separates the elemental concerns of an RL/control task.
    By separating each concern, we can experiment with different combinations of these elements.
    For example, multiple reward structures can be easily tested on the same process.
    """
    
    def __init__(self,
                 process: Process,
                 observer: Observer,
                 terminator: Terminator,
                 rewarder: Rewarder,
                 action_transforms: [ActionTransform] = []):
        """
        :param process: The process under control
        :param rewarder: Rewarder that gets a reward given a process state
        :param terminator: Terminator that determines when the process has reached a terminal state
        :param observer: Observer that extracts, formats, and returns an observation from the process
        :param action_transform: converts the agent's action into the process's input format
        """
        super().__init__()
        
        self.process: Process = process
        self.view: Optional[View] = None
        self.observer: Observer = observer
        self.terminator: Terminator = terminator
        self.rewarder: Rewarder = rewarder
        self.action_transforms: [ActionTransform] = action_transforms
        
        self.components: [FactoredGymComponent] = [self.process, self.observer, self.terminator, self.rewarder]
        self.components.extend(self.action_transforms)
        
        self.action_space = self.process.get_action_space()
        for transform in reversed(self.action_transforms):
            self.action_space = transform.get_action_space(self.process, self.action_space)
        
        self.observation_space = self.observer.get_observation_space(self.process)
        self.action = None
    
    def reset(self) -> Any:
        """
        Resets the process to an initial state. Calls self.process.reset()
        :return an initial observation

        """
        for component in self.components:
            component.reset(self.process)
        return self.observer.get_observation(self.process)
    
    def step(self, action) -> Tuple[Any, float, bool, Any]:
        """
        Advances the process one timestep while applying the given action.
        :param action: action to take at this timestep
        :return tuple of (observation, reward, terminated, extra)
        """
        transformed_action = self.compute_transformed_action(action)
        return self.step_with_transformed_action(transformed_action)
    
    def render(self, mode='human', close=False) -> None:
        """
        Renders the current process state in some visual way.
        Views are lazily initialized by calling self.process.make_view()
        You may close a view with the close_view() method.
        :param mode: OpenAI Gym mode parameter
        :param close: True if you want to close the view (same as calling close_view())
        :return:
        """
        if close:
            self.close_view()
            return
        
        if self.view is None:
            self.view = self.process.make_view(mode)
            self.components.append(self.view)
        
        for component in self.components:
            component.begin_render(self.process, self.view)
        
        # self.process, self.observer, self.terminator, self.rewarder, self.view
        for component in self.components:
            component.render(self.process, self.view)
    
    def close(self) -> None:
        """
        Closes the gym.
        """
        self.close_view()  # close view first (if there is one)
        
        for component in reversed(self.components):
            component.close(self.process)
    
    def close_view(self) -> None:
        """
        Closes the view if it is open
        :return:
        """
        if self.view is None:
            return
        self.view.close(self.process)
        del self.components[-1]
        self.view = None
    
    def compute_transformed_action(self, action) -> any:
        for transform in self.action_transforms:
            action = transform.transform_action(self.process, action)
        return action
    
    def step_with_transformed_action(self, transformed_action: any) -> Tuple[Any, float, bool, Any]:
        self.action = transformed_action  # saves action for reference
        extra = self.process.step(transformed_action)
        observation = self.observer.get_observation(self.process)
        terminated = self.terminator.is_terminal(self.process, observation)
        reward = self.rewarder.get_reward(self.process, observation, terminated)
        return observation, reward, terminated, extra
