from BearRobot.utils.evaluation.mp_libero_eval import LIBEROEval
from BearRobot.utils.logger.tb_log import TensorBoardLogger
from BearRobot.Agent import build_visual_diffsuion, build_minibc

def main():
    name = '0622test06'

    ckpt_path = f'/home/fishyu/zh1hao_space/experiments/libero/libero_goal/minibc/{name}/latest.pth'
    statistic_path = f'/home/fishyu/zh1hao_space/experiments/libero/libero_goal/minibc/{name}/statistics.json'

    policy = build_minibc(ckpt_path, statistic_path)
    evaluator = LIBEROEval(task_suite_name='libero_goal', data_statistics=None, eval_horizon=50, num_episodes=3)

    evaluator.eval_episodes(policy, 0, save_path=f'/home/fishyu/zh1hao_space/experiments/libero/libero_goal/minibc/{name}/')

if __name__ == '__main__':
    main()