from . import ac
from . import dcvtd_ac
from . import q_learning
from . import dcvtd_q_learning

AC = ac.ActorCritic
DCVTD_AC = dcvtd_ac.DCVTD_ActorCritic
IL = q_learning.DQN
DCVTD_Q = dcvtd_q_learning.DCVTD_DQN



def spawn_ai(algo_name, sess, env, handle, human_name, max_steps):
    if algo_name == 'ac':
        model = AC(sess, human_name, handle, env)
    elif algo_name == 'ac_dcvtd':
        model = DCVTD_AC(sess, human_name, handle, env)
    elif algo_name == 'il':
        model = IL(sess, human_name, handle, env, max_steps, memory_size=80000)
    elif algo_name == 'il_dcvtd':
        model = DCVTD_Q(sess, human_name, handle, env, max_steps, memory_size=80000)
    return model