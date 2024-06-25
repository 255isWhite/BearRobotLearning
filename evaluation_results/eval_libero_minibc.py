from BearRobot.utils.evaluation.mp_libero_eval import LIBEROEval
from BearRobot.utils.logger.tb_log import TensorBoardLogger
from BearRobot.Agent import build_visual_diffsuion, build_minibc

def main():
# logger = TensorBoardLogger(project_name='test', run_name='test', )
    ckpt_path = '/home/fishyu/zh1hao_space/experiments/libero/libero_goal/minibc/0622test04/latest.pth'
    statistic_path = '/home/fishyu/zh1hao_space/experiments/libero/libero_goal/minibc/0622test04/statistics.json'

    policy = build_minibc(ckpt_path, statistic_path)
    evaluator = LIBEROEval(task_suite_name='libero_goal', data_statistics=None, eval_horizon=300, num_episodes=10)

    evaluator.eval_episodes(policy, 0, save_path='/home/fishyu/zh1hao_space/experiments/libero/libero_goal/minibc/0622test04/')

if __name__ == '__main__':
    main()