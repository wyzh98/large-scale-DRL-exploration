import ray
import torch

from model import PolicyNet
from env import Env
from agent import Agent
from utils import *
from parameter import *

# Test configuration
NUM_TEST = 100
NUM_META_AGENT = 10
SAVE_GIFS = False

if SAVE_GIFS:
    if not os.path.exists(gifs_path):
        os.makedirs(gifs_path, exist_ok=True)


def run_test():
    device = torch.device('cpu')
    policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)

    print(f"Testing on {device}, model: {model_path}, num of tests: {NUM_TEST}")
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(f"{model_path}/checkpoint.pth", map_location=device)
    policy_net.load_state_dict(checkpoint["policy_model"])

    # launch meta agents
    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    weights = policy_net.state_dict()
    curr_test = 0

    travel_dist = []
    explored_rate = []
    success_rate = []

    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        if curr_test >= NUM_TEST:
            break
        job_list.append(meta_agent.job.remote(weights, curr_test))
        curr_test += 1

    try:
        while len(travel_dist) < NUM_TEST:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)

            for metrics, info in done_jobs:
                travel_dist.append(metrics["travel_dist"])
                explored_rate.append(metrics["explored_rate"])
                success_rate.append(metrics["success_rate"])

                if curr_test < NUM_TEST:
                    job_list.append(meta_agents[info["id"].__int__()].job.remote(weights, curr_test))
                    curr_test += 1

            if len(travel_dist) >= NUM_TEST:
                break

        print("=====================================")
        print("| Test:", FOLDER_NAME)
        print("| Total test episodes:", NUM_TEST)
        print("| Average success rate:", np.array(success_rate, dtype=float).mean())
        print("| Average explored rate:", np.array(explored_rate, dtype=float).mean())
        print("| Average travel distance:", np.array(travel_dist, dtype=float).mean())

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


class TestWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device="cpu", save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image, test=True)
        self.robot = Agent(policy_net, device=self.device, plot=self.save_image)

        self.perf_metrics = dict()

    def run_episode(self):
        done = False

        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        observation = self.robot.get_observation()

        if self.save_image:
            self.robot.plot_env()
            self.env.plot_env(0)

        for step in range(MAX_EPISODE_STEP):
            next_location, action_index = self.robot.select_next_waypoint(observation)
            self.env.step(next_location)
            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
            if self.robot.utility.sum() == 0 or self.env.explored_rate > 0.9999:
                done = True
            observation = self.robot.get_observation()
            if self.save_image:
                self.robot.plot_env()
                self.env.plot_env(step + 1)

            if done:
                break

        # save metrics
        self.perf_metrics["travel_dist"] = self.env.travel_dist
        self.perf_metrics["explored_rate"] = self.env.explored_rate
        self.perf_metrics["success_rate"] = done

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)


@ray.remote(num_cpus=1)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cpu')
        self.worker = None
        self.network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
        self.network.to(self.device)

    def set_weights(self, weights):
        self.network.load_state_dict(weights)

    def do_job(self, episode_number):
        save_img = False
        self.worker = TestWorker(self.meta_agent_id, self.network, episode_number, device=self.device, save_image=save_img)
        self.worker.run_episode()
        perf_metrics = self.worker.perf_metrics
        return perf_metrics

    def job(self, weights, episode_number):
        print(f"starting episode {episode_number} on metaAgent {self.meta_agent_id}")

        self.set_weights(weights)
        metrics = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return metrics, info


if __name__ == '__main__':
    ray.init()
    run_test()