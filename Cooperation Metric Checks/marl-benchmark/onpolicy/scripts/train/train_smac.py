#!/usr/bin/env python
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
import argparse
import numpy as np
import setproctitle
import torch
import wandb
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv

"""Train script for SMAC."""


def parse_smacv2_distribution(args):
    units = args.units.split("v")
    distribution_config = {
        "n_units": int(units[0]),
        "n_enemies": int(units[1]),
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "map_x": 32,
            "map_y": 32,
        },
    }
    if "protoss" in args.map_name:
        distribution_config["team_gen"] = {
            "dist_type": "weighted_teams",
            "unit_types": ["stalker", "zealot", "colossus"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        }
    elif "zerg" in args.map_name:
        distribution_config["team_gen"] = {
            "dist_type": "weighted_teams",
            "unit_types": ["zergling", "baneling", "hydralisk"],
            "weights": [0.45, 0.1, 0.45],
            "observe": True,
        }
    elif "terran" in args.map_name:
        distribution_config["team_gen"] = {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        }
    return distribution_config


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env

                env = StarCraft2Env(all_args)
            elif all_args.env_name == "StarCraft2v2":
                from onpolicy.envs.starcraft2.SMACv2_modified import SMACv2

                env = SMACv2(
                    capability_config=parse_smacv2_distribution(all_args),
                    map_name=all_args.map_name,
                )
            elif all_args.env_name == "SMAC":
                from onpolicy.envs.starcraft2.SMAC import SMAC

                env = SMAC(map_name=all_args.map_name)
            elif all_args.env_name == "SMACv2":
                from onpolicy.envs.starcraft2.SMACv2 import SMACv2

                env = SMACv2(
                    capability_config=parse_smacv2_distribution(all_args),
                    map_name=all_args.map_name,
                )
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
        )


def make_single_eval_env(all_args):
    """only make one environment"""
    print("making single eval env: ", all_args.env_name)
    rank = 0
    if all_args.env_name == "StarCraft2":
        from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env

        env = StarCraft2Env(all_args)
    elif all_args.env_name == "StarCraft2v2":
        from onpolicy.envs.starcraft2.SMACv2_modified import SMACv2

        env = SMACv2(
            capability_config=parse_smacv2_distribution(all_args),
            map_name=all_args.map_name,
        )
    elif all_args.env_name == "SMAC":
        from onpolicy.envs.starcraft2.SMAC import SMAC

        env = SMAC(map_name=all_args.map_name)
    elif all_args.env_name == "SMACv2":
        from onpolicy.envs.starcraft2.SMACv2 import SMACv2

        env = SMACv2(
            capability_config=parse_smacv2_distribution(all_args),
            map_name=all_args.map_name,
        )
    else:
        print("Can not support the " + all_args.env_name + "environment.")
        raise NotImplementedError
    env.seed(all_args.seed * 50000 + rank * 10000)

    # return ShareDummyVecEnv([env])

    return env


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env

                env = StarCraft2Env(all_args)
            elif all_args.env_name == "StarCraft2v2":
                from onpolicy.envs.starcraft2.SMACv2_modified import SMACv2

                env = SMACv2(
                    capability_config=parse_smacv2_distribution(all_args),
                    map_name=all_args.map_name,
                )
            elif all_args.env_name == "SMAC":
                from onpolicy.envs.starcraft2.SMAC import SMAC

                env = SMAC(map_name=all_args.map_name)
            elif all_args.env_name == "SMACv2":
                from onpolicy.envs.starcraft2.SMACv2 import SMACv2

                env = SMACv2(
                    capability_config=parse_smacv2_distribution(all_args),
                    map_name=all_args.map_name,
                )
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
        )


def parse_args(args, parser):
    parser.add_argument(
        "--map_name", type=str, default="4m", help="Which smac map to run on"
    )
    parser.add_argument("--units", type=str, default="10v10")  # for smac v2
    parser.add_argument("--add_move_state", action="store_true", default=False)
    parser.add_argument("--add_local_obs", action="store_true", default=False)
    parser.add_argument("--add_distance_state", action="store_true", default=False)
    parser.add_argument("--add_enemy_action_state", action="store_true", default=False)
    parser.add_argument("--add_agent_id", action="store_true", default=False)
    parser.add_argument("--add_visible_state", action="store_true", default=False)
    parser.add_argument("--add_xy_state", action="store_true", default=False)
    parser.add_argument("--use_state_agent", action="store_false", default=True)
    parser.add_argument("--use_mustalive", action="store_false", default=True)
    parser.add_argument("--add_center_xy", action="store_false", default=True)
    parser.add_argument(
        "--teamB_map_name",
        type=str,
        default="4m",
        help="Which smac map params to use for second team",
    )

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    print(all_args)

    assert (
        all_args.group_social_influence
        in [
            "OnlySocialRwd",
            "CombineSocialGameRwd",
            "OnlyGameRwd",
            "CombineSocialGameRwd_withPenalty",
            "OnlySocialRwd_withPenalty",
            "MultiplyGameSocialRwd",
            "OnlyConditionalSocial",
            "RandomRewardsUniform",
            "RandomRewardGaussian",
            "RandomRewardMimic",
            "OnlyRandomRewardsUniform",
        ]
    ), "Must specify what reward to use [OnlySocialRwd, CombineSocialGameRwd, OnlyGameRwd, CombineSocialGameRwd_withPenalty, OnlySocialRwd_withPenalty, MultiplyGameSocialRwd, OnlyConditionalSocial, RandomRewardsUniform, RandomRewardGaussian, RandomRewardMimic, OnlyRandomRewardsUniform]"

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif (
        all_args.algorithm_name == "mappo"
        or all_args.algorithm_name == "mat"
        or all_args.algorithm_name == "mat_dec"
    ):
        print(
            "u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False"
        )
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    elif all_args.algorithm_name == "happo" or all_args.algorithm_name == "hatrpo":
        # can or cannot use recurrent network?
        print("using", all_args.algorithm_name, "without recurrent network")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "qmix":
        print(
            "u are choosing to use qmix, we set use_centralized_V & use_recurrent_policy & use_naive_recurrent_policy to be False"
        )
        all_args.use_centralized_V = False
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
        # TODO add the qmix params here and use the qmix network
    else:
        raise NotImplementedError

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    prepend = (
        "two_social_influence_comparison_"
        if all_args.evaluate_two_social_influence
        else ""
    )

    if all_args.model_dir is None:
        run_dir = (
            Path(
                os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
                + f"/{prepend}results"
            )
            / all_args.env_name
            / all_args.map_name
            / all_args.algorithm_name
            / all_args.experiment_name
        )
    else:
        # split for just the model name
        modelDirName = all_args.model_dir.split("/")[-1]
        run_dir = (
            Path(
                os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
                + f"/{prepend}loaded_results"
            )
            / all_args.env_name
            / all_args.map_name
            / all_args.algorithm_name
            / all_args.experiment_name
            / modelDirName
        )

    differentOriginEnvTeams = all_args.map_name != all_args.teamB_map_name
    if differentOriginEnvTeams:
        if all_args.policy_model_dir_teamB is None:
            policy_model_dir_teamB = "NewTeamB"
        else:
            policy_model_dir_teamBList = all_args.policy_model_dir_teamB.split("/")
            WhichRun = policy_model_dir_teamBList[-2]
            WhichPolicy = (
                policy_model_dir_teamBList[-1]
                .replace("_Episode", "")
                .replace(".0M", "M")
            )
            policy_model_dir_teamB = WhichRun + "_" + WhichPolicy

        policy_model_dir_teamAList = all_args.policy_model_dir_teamA.split("/")[-1]
        WhichRun = policy_model_dir_teamAList[-2]
        WhichPolicy = (
            policy_model_dir_teamAList[-1].replace("_Episode", "").replace(".0M", "M")
        )
        policy_model_dir_teamA = WhichRun + "_" + WhichPolicy

        adhocVs = policy_model_dir_teamA + "_vs_" + policy_model_dir_teamB
        shortenedAdhocVs = adhocVs.replace("SMAC-", "").replace("MAPPO-", "")

        if all_args.no_train_policy:
            Training = "NoTrainPolicy"
        else:
            Training = "TrainPolicy"

        if all_args.no_train_si:
            Training += "_NoTrainSI"
        else:
            Training += "_TrainSI"

        if all_args.no_train_policy and all_args.no_train_si:
            Training = "EvalOnly"

        if not all_args.no_train_policy:
            Training += "_"
            Training += (
                all_args.group_social_influence
                + "_("
                + str(all_args.group_social_influence_factor)
                + ")"
            )
            print("Reward Type: ", all_args.group_social_influence)
            print("Reward Factor: ", all_args.group_social_influence_factor)
        run_dir = (
            Path(
                os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
                + f"/{prepend}adhoc_results"
            )
            / all_args.env_name
            / all_args.map_name
            / all_args.algorithm_name
            / all_args.experiment_name
            / shortenedAdhocVs
            / Training
        )

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # create for gifs even if wandb
    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    tags = ["cooperation-metric-check", all_args.env_name]
    if all_args.model_dir is not None:
        tags += ["LoadedPolicyModel"]

    if all_args.si_model_dir_teamB is not None:
        # Get model checkpoints
        # find the number between social_influence_ts and .0M
        si_A_checkpoint = (
            all_args.si_model_dir_teamA.split("/")[-1]
            .replace("social_influence_ts", "")
            .replace(".0M", "M")
        )
        si_B_checkpoint = (
            all_args.si_model_dir_teamB.split("/")[-1]
            .replace("social_influence_ts", "")
            .replace(".0M", "M")
        )

        # Check if the number is the same
        if si_A_checkpoint != si_B_checkpoint:
            raise ValueError("Social influence model checkpoints do not match")

        # Add the social influence checkpoint to the tags
        all_args.si_model_checkpoint = si_A_checkpoint

        policy_A_checkpoint = (
            all_args.policy_model_dir_teamA.split("/")[-1]
            .replace("Policy_Episode", "")
            .replace(".0M", "M")
        )
        policy_B_checkpoint = (
            all_args.policy_model_dir_teamB.split("/")[-1]
            .replace("Policy_Episode", "")
            .replace(".0M", "M")
        )

        # Check if the number is the same
        if policy_A_checkpoint != policy_B_checkpoint:
            raise ValueError("Policy model checkpoints do not match")

        # Add the policy checkpoint to the tags
        all_args.policy_model_checkpoint = policy_A_checkpoint

        all_args.checkpoints_statement = (
            f"{policy_A_checkpoint}Policy_{si_A_checkpoint}SI"
        )

        # tags += [f"LoadSI-{si_A_checkpoint}M", f"LoadPolicy-{policy_A_checkpoint}M"]
    if all_args.use_wandb:
        wandb.login(key="83385f5ec22fac4705b677a03b63a470e70d61da")
        try:
            run = wandb.init(
                config=all_args,
                project="Multi Agent Joint Skills",
                entity="moms-2023",
                tags=tags,
                notes=socket.gethostname(),
                name=str(all_args.algorithm_name)
                + "_"
                + str(all_args.experiment_name)
                + "_"
                + str(all_args.units)
                + "_seed"
                + str(all_args.seed)
                + curr_run,
                #  group=all_args.map_name,
                dir=str(run_dir),
                job_type="training",
                reinit=True,
            )
        except Exception as e:
            print(e)
            print("Logging to wandb failed, logging to no project and entity instead")
            run = wandb.init(
                config=all_args,
                # project="Multi Agent Joint Skills",
                # entity="moms-2023",
                tags=tags,
                notes=socket.gethostname(),
                name=str(all_args.algorithm_name)
                + "_"
                + str(all_args.experiment_name)
                + "_"
                + str(all_args.units)
                + "_seed"
                + str(all_args.seed)
                + curr_run,
                dir=str(run_dir),
            )

        all_args = wandb.config  # for wandb sweep
        
        

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    single_eval_envs = make_single_eval_env(all_args)

    if all_args.env_name == "SMAC":
        from smac.env.starcraft2.maps import get_map_params

        num_agents = get_map_params(all_args.map_name)["n_agents"]
    elif all_args.env_name == "StarCraft2":
        from onpolicy.envs.starcraft2.smac_maps import get_map_params

        num_agents = get_map_params(all_args.map_name)["n_agents"]
    elif all_args.env_name == "SMACv2" or all_args.env_name == "StarCraft2v2":
        from smacv2.env.starcraft2.maps import get_map_params

        num_agents = parse_smacv2_distribution(all_args)["n_units"]

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "single_eval_envs": single_eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    differentOriginEnvTeams = all_args.map_name != all_args.teamB_map_name
    teamB_map_name = all_args.teamB_map_name
    config["differentOriginEnvTeams"] = differentOriginEnvTeams
    if differentOriginEnvTeams:
        if isinstance(all_args, dict):
            teamBArgs = all_args.copy()
        elif isinstance(all_args, SimpleNamespace) or isinstance(all_args, argparse.Namespace):
            teamBArgs = vars(all_args).copy() # if wandb is off?
        else:
            teamBArgs = dict(all_args) # if wandb is on?
        teamBArgs["map_name"] = teamB_map_name
        TeamBInput = SimpleNamespace(**teamBArgs)
        # Have to set up team B envs - even if we are not using them because got to get variables
        teamB_envs = make_train_env(TeamBInput)
        teamB_eval_envs = make_eval_env(TeamBInput) if all_args.use_eval else None
        teamB_single_eval_envs = make_single_eval_env(TeamBInput)

        config["teamB_envs"] = teamB_envs
        # Does not seem to be required yet
        # config["teamB_eval_envs"] = teamB_eval_envs
        # config["teamB_single_eval_envs"] = teamB_single_eval_envs
        config["teamB_map_name"] = all_args.teamB_map_name

    # run experiments
    if all_args.share_policy:
        if all_args.evaluate_two_social_influence and all_args.no_train_policy:
            from onpolicy.runner.shared.smac_runner_social_influence_evaluation import (
                SMACRunner as Runner,
            )
        else:
            from onpolicy.runner.shared.smac_runner import (
                SMACRunner as Runner,
            )  # This is true for MAPPO
    else:
        from onpolicy.runner.separated.smac_runner import SMACRunner as Runner

    if all_args.algorithm_name == "happo" or all_args.algorithm_name == "hatrpo":
        from onpolicy.runner.separated.smac_runner import SMACRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.summary_writer.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.summary_writer.close()


if __name__ == "__main__":
    main(sys.argv[1:])
if __name__ == "__main__":
    main(sys.argv[1:])
    main(sys.argv[1:])
    main(sys.argv[1:])
    main(sys.argv[1:])
    main(sys.argv[1:])
    main(sys.argv[1:])
    main(sys.argv[1:])
    main(sys.argv[1:])
    main(sys.argv[1:])
    main(sys.argv[1:])
    main(sys.argv[1:])
    main(sys.argv[1:])
    main(sys.argv[1:])
    main(sys.argv[1:])
    main(sys.argv[1:])
    main(sys.argv[1:])
