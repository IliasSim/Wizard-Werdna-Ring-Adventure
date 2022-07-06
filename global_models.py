import numpy as np
import tensorflow as tf
from os.path import exists
input1 = (1, 1, 148, 148, 3)
input2 = (1,1,10)
input3 = (1,1,125)
from playerActorLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents import actor,critic
global_actor = actor()
global_critic = critic()
global_actor(tf.convert_to_tensor(np.random.random((input1)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input2)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input3)), dtype=tf.float32))
global_critic(tf.convert_to_tensor(np.random.random((input1)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input2)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input3)), dtype=tf.float32))
if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents\ actor_model2.data-00000-of-00001'):
        print("actor model is loaded")
        global_actor.load_weights('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents\ actor_model2')
if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents\ critic_model2.data-00000-of-00001'):
        print("critic model is loaded")
        global_actor.load_weights('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents\ critic_model2')