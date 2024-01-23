try:
    from tblite import interface
except:
    pass

import argparse
import os
import pickle
from pathlib import Path

import getdist.plots as gdplt
import numpy as np
import torch
from cobaya.run import run
from getdist.mcsamples import MCSamplesFromCobaya
from gflownet.envs.conformers.conformer import Conformer
from gflownet.proxy.conformers.tblite import TBLiteMoleculeEnergy
from gflownet.proxy.conformers.torchani import TorchANIMoleculeEnergy
from gflownet.proxy.conformers.xtb import XTBMoleculeEnergy
from scipy import stats


def convert_to_numpy_if_needed(array):
    if torch.is_tensor(array):
        return array.cpu().detach().numpy()
    else:
        return array


def main(args):
    if args.proxy_name == "torchani":
        proxy_class = TorchANIMoleculeEnergy
    elif args.proxy_name == "tblite":
        proxy_class = TBLiteMoleculeEnergy
    elif args.proxy_name == "xtb":
        proxy_class = XTBMoleculeEnergy

    # Leave as is
    DEVICE = "cpu"
    FLOAT_PRECISION = 32
    REWARD_FUNC = "boltzmann"
    REWARD_BETA = 32

    # output dir
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        os.mkdir(output_dir)

    for smile in args.ids:
        # Change n_torsion_angles
        env = Conformer(
            smiles=int(smile),
            n_torsion_angles=-1,
            reward_func=REWARD_FUNC,
            reward_beta=REWARD_BETA,
        )

        ndims = len(env.torsion_angles)

        proxy = proxy_class(device=DEVICE, float_precision=FLOAT_PRECISION)
        proxy.setup(env)

        print(f"Sampling for {ndims} dimensions with {args.proxy_name} proxy")

        if ndims == 2:

            def reward(p0, p1):
                batch = np.concatenate([[p0], [p1]]).reshape(1, -1)
                batch = np.concatenate([batch, np.zeros((batch.shape[0], 1))], axis=-1)

                proxy_batch = env.statebatch2proxy(batch)
                energies = proxy(env.statebatch2proxy(batch))
                rewards = env.proxy2reward(-energies)
                rewards = convert_to_numpy_if_needed(rewards)
                return np.log(rewards)

        elif ndims == 3:

            def reward(p0, p1, p2):
                batch = np.concatenate([[p0], [p1], [p2]]).reshape(1, -1)
                batch = np.concatenate([batch, np.zeros((batch.shape[0], 1))], axis=-1)

                proxy_batch = env.statebatch2proxy(batch)
                energies = proxy(env.statebatch2proxy(batch))
                rewards = env.proxy2reward(-energies)
                rewards = convert_to_numpy_if_needed(rewards)
                return np.log(rewards)

        elif ndims == 4:

            def reward(p0, p1, p2, p3):
                batch = np.concatenate([[p0], [p1], [p2], [p3]]).reshape(1, -1)
                batch = np.concatenate([batch, np.zeros((batch.shape[0], 1))], axis=-1)

                proxy_batch = env.statebatch2proxy(batch)
                energies = proxy(env.statebatch2proxy(batch))
                rewards = env.proxy2reward(-energies)
                rewards = convert_to_numpy_if_needed(rewards)
                return np.log(rewards)

        if ndims == 5:

            def reward(p0, p1, p2, p3, p4):
                batch = np.concatenate([[p0], [p1], [p2], [p3], [p4]]).reshape(1, -1)
                batch = np.concatenate([batch, np.zeros((batch.shape[0], 1))], axis=-1)

                proxy_batch = env.statebatch2proxy(batch)
                energies = proxy(env.statebatch2proxy(batch))
                rewards = env.proxy2reward(-energies)
                rewards = convert_to_numpy_if_needed(rewards)
                return np.log(rewards)

        if ndims == 6:

            def reward(p0, p1, p2, p3, p4, p5):
                batch = np.concatenate([[p0], [p1], [p2], [p3], [p4], [p5]]).reshape(
                    1, -1
                )
                batch = np.concatenate([batch, np.zeros((batch.shape[0], 1))], axis=-1)

                proxy_batch = env.statebatch2proxy(batch)
                energies = proxy(env.statebatch2proxy(batch))
                rewards = env.proxy2reward(-energies)
                rewards = convert_to_numpy_if_needed(rewards)
                return np.log(rewards)

        elif ndims == 7:

            def reward(p0, p1, p2, p3, p4, p5, p6):
                batch = np.concatenate(
                    [[p0], [p1], [p2], [p2], [p3], [p4], [p5], [p6]]
                ).reshape(1, -1)
                batch = np.concatenate([batch, np.zeros((batch.shape[0], 1))], axis=-1)

                proxy_batch = env.statebatch2proxy(batch)
                energies = proxy(env.statebatch2proxy(batch))
                rewards = env.proxy2reward(-energies)
                rewards = convert_to_numpy_if_needed(rewards)
                return np.log(rewards)

        elif ndims == 8:

            def reward(p0, p1, p2, p3, p4, p5, p6, p7):
                batch = np.concatenate(
                    [[p0], [p1], [p2], [p3], [p4], [p5], [p6], [p7]]
                ).reshape(1, -1)
                batch = np.concatenate([batch, np.zeros((batch.shape[0], 1))], axis=-1)

                proxy_batch = env.statebatch2proxy(batch)
                energies = proxy(env.statebatch2proxy(batch))
                rewards = env.proxy2reward(-energies)
                rewards = convert_to_numpy_if_needed(rewards)
                return np.log(rewards)

        elif ndims == 9:

            def reward(p0, p1, p2, p3, p4, p5, p6, p7, p8):
                batch = np.concatenate(
                    [[p0], [p1], [p2], [p3], [p4], [p5], [p6], [p7], [p8]]
                ).reshape(1, -1)
                batch = np.concatenate([batch, np.zeros((batch.shape[0], 1))], axis=-1)

                proxy_batch = env.statebatch2proxy(batch)
                energies = proxy(env.statebatch2proxy(batch))
                rewards = env.proxy2reward(-energies)
                rewards = convert_to_numpy_if_needed(rewards)
                return np.log(rewards)

        elif ndims == 10:

            def reward(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9):
                batch = np.concatenate(
                    [[p0], [p1], [p2], [p3], [p4], [p5], [p6], [p7], [p8], [p9]]
                ).reshape(1, -1)
                batch = np.concatenate([batch, np.zeros((batch.shape[0], 1))], axis=-1)

                proxy_batch = env.statebatch2proxy(batch)
                energies = proxy(env.statebatch2proxy(batch))
                rewards = env.proxy2reward(-energies)
                rewards = convert_to_numpy_if_needed(rewards)
                return np.log(rewards)

        elif ndims == 11:

            def reward(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10):
                batch = np.concatenate(
                    [[p0], [p1], [p2], [p3], [p4], [p5], [p6], [p7], [p8], [p9], [p10]]
                ).reshape(1, -1)
                batch = np.concatenate([batch, np.zeros((batch.shape[0], 1))], axis=-1)

                proxy_batch = env.statebatch2proxy(batch)
                energies = proxy(env.statebatch2proxy(batch))
                rewards = env.proxy2reward(-energies)
                rewards = convert_to_numpy_if_needed(rewards)
                return np.log(rewards)

        elif ndims == 12:

            def reward(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11):
                batch = np.concatenate(
                    [
                        [p0],
                        [p1],
                        [p2],
                        [p3],
                        [p4],
                        [p5],
                        [p6],
                        [p7],
                        [p8],
                        [p9],
                        [p10],
                        [p11],
                    ]
                ).reshape(1, -1)
                batch = np.concatenate([batch, np.zeros((batch.shape[0], 1))], axis=-1)

                proxy_batch = env.statebatch2proxy(batch)
                energies = proxy(env.statebatch2proxy(batch))
                rewards = env.proxy2reward(-energies)
                rewards = convert_to_numpy_if_needed(rewards)
                return np.log(rewards)

        info = {"likelihood": {"reward": reward}}

        info["params"] = {
            f"p{i}": {
                "prior": {"min": 0, "max": 2 * np.pi},
                "ref": np.pi,
                "proposal": 0.01,
            }
            for i in range(ndims)
        }

        Rminus1_stop = args.rminus1_stop
        info["sampler"] = {"mcmc": {"Rminus1_stop": Rminus1_stop, "max_tries": 1000}}

        updated_info, sampler = run(info)

        gdsamples = MCSamplesFromCobaya(updated_info, sampler.products()["sample"])

        def get_energy(batch):
            batch = np.concatenate([batch, np.zeros((batch.shape[0], 1))], axis=-1)

            proxy_batch = env.statebatch2proxy(batch)
            energies = proxy(env.statebatch2proxy(batch))
            return energies

        if gdsamples.samples.shape[0] >= 1000:
            npars = len(info["params"])
            dct = {
                "x": gdsamples.samples[-1000:, :npars]
            }  # , "energy": np.exp(gdsamples.loglikes[-10000:])}

            dct["energy"] = get_energy(gdsamples.samples[-1000:, :npars])

            dct["conformer"] = [
                env.set_conformer(state).rdk_mol
                for state in gdsamples.samples[-1000:, :npars]
            ]
            pickle.dump(
                dct,
                open(
                    output_dir / f"conformers_mcmc_smile{smile}_{args.proxy_name}.pkl",
                    "wb",
                ),
            )
            print(f"Finished smile {smile} (dimensions {ndims})")

        else:
            print(f"Not enough samples for smile {smile}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", required=True, type=int)
    parser.add_argument("--output_dir", type=str, default="./mcmc_outputs/")
    parser.add_argument("--proxy_name", type=str, default="torchani")
    parser.add_argument("--rminus1_stop", type=float, default=0.05)
    args = parser.parse_args()
    main(args)
