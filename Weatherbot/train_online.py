#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import rasa_core

from rasa_core.run import serve_application
from rasa_core import config
from rasa_core.agent import Agent
#from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.interpreter import RasaNLUInterpreter

logger = logging.getLogger(__name__)


def run_weather_online(input_channel, interpreter,
                          domain_file="weather_domain.yml",
                          training_data_file='data/stories.md'):
            agent = Agent('weather_domain.yml', policies = [MemoizationPolicy(max_history=2), KerasPolicy(max_history=4)])
            data = agent.load_data(training_data_file)
            agent.train(data)
            rasa_core.run.serve_application(agent ,channel='cmdline')
            #interactive.run_interactive_learning(agent, training_data_file)
            return agent 

if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    nlu_interpreter = RasaNLUInterpreter('./models/nlu/default/weatherbot')
    #run_weather_online(ConsoleInputChannel(), nlu_interpreter)
    run_weather_online('cmdline', nlu_interpreter)



# In[ ]:




