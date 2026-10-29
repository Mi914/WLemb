from pathlib import Path
import shlex
import pandas as pd
from spacegraph_codebase.trainer import make_args_parser, make_args_combine
import re
from pathlib import Path
from spacegraph_codebase.utils import *
from spacegraph_codebase.Place2Vec.cur_data_utils import load_pointset
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.collections import PatchCollection
import json
from PIL import Image



def extract_test_metrics(log_file):
    text = Path(log_file).read_text(
        encoding="utf-8", errors="replace"
    )

    pattern = (
        r"Iter:\s*(\d+);\s*(\d+)\s+Neg,\s*"
        r"(Test|Validate)\s+MRR:\s*([\d.]+),\s*"
        r"HIT@1:\s*([\d.]+),\s*"
        r"HIT@5:\s*([\d.]+),\s*"
        r"HIT@10:\s*([\d.]+)"
    )

    matches = re.findall(pattern, text)

    result = {}

    for iteration, neg, split, mrr, hit1, hit5, hit10 in matches:
        prefix = "valid" if split == "Validate" else "test"

        result[f"{prefix}_mrr_{neg}neg"] = float(mrr)
        result[f"{prefix}_hit1_{neg}neg"] = float(hit1)
        result[f"{prefix}_hit5_{neg}neg"] = float(hit5)
        result[f"{prefix}_hit10_{neg}neg"] = float(hit10)
        result[f"{prefix}_iter_{neg}neg"] = int(iteration)

    return result

def extract_stg_metrics(log_file):
    text = Path(log_file).read_text(
        encoding="utf-8", errors="replace"
    )

    result = {}

    # STGの選択基底数・確率
    pattern_stg = (
        r"STG:\s*total=(\d+),\s*"
        r"selected\(p>0\.5\)=(\d+),\s*"
        r"expected_active=([-\d.]+),\s*"
        r"mean_prob=([-\d.]+)"
    )

    matches = re.findall(pattern_stg, text)

    if matches:
        total, selected, expected, mean_prob = matches[-1]

        result.update({
            "stg_total": int(total),
            "stg_selected": int(selected),
            "stg_expected_active": float(expected),
            "stg_mean_prob": float(mean_prob),
            "stg_selected_ratio": int(selected) / int(total),
        })

    # mu の統計情報
    pattern_mu = (
        r"mu_min\s*=\s*([-\d.]+),\s*"
        r"mu_max\s*=\s*([-\d.]+),\s*"
        r"mu_mean\s*=\s*([-\d.]+),\s*"
        r"\(mu<=0\)\.sum\(\)\s*=\s*(\d+)"
    )

    matches = re.findall(pattern_mu, text)

    if matches:
        mu_min, mu_max, mu_mean, mu_nonpositive = matches[-1]

        result.update({
            "mu_min": float(mu_min),
            "mu_max": float(mu_max),
            "mu_mean": float(mu_mean),
            "mu_nonpositive": int(mu_nonpositive),
        })

    return result

def make_results_df(exp_dir: Path):
    parser = make_args_parser()

    scripts_dir = Path(exp_dir / "scripts")
    results_dir = Path(exp_dir / "results")

    records = []

    for sh_file in sorted(scripts_dir.glob("*.sh")):
        command = sh_file.read_text(encoding="utf-8")

        # 行末の \ による改行を結合
        command = command.replace("\\\n", " ")

        # コメント・空行を除外し、単語に分割
        tokens = shlex.split(command, comments=True)

        # trainモジュールの位置を探す
        module_name = "spacegraph_codebase.Place2Vec.train"

        if module_name not in tokens:
            print(f"SKIP: {sh_file.name}")
            continue

        start = tokens.index(module_name) + 1
        args_list = tokens[start:]

        args = parser.parse_args(args_list)
        args_com = make_args_combine(args)

        log_file = Path(results_dir) / f"{args_com.lstrip('/')}.log"
        
        record = vars(args).copy()
        record["script"] = sh_file.name
        record["args_com"] = args_com
        record["log_file"] = str(log_file)
        record["log_exists"] = log_file.exists()

        records.append(record)

    df = pd.DataFrame(records)

    print(f"実験数: {len(df)}")
    print(f"ログあり: {df['log_exists'].sum()}")

    for i, row in df.iterrows():
        log_file = Path(row["log_file"])

        if not log_file.is_file():
            continue

        metrics = extract_test_metrics(log_file)

        for key, value in metrics.items():
            df.loc[i, key] = value

        metrics = extract_stg_metrics(log_file)

        for key, value in metrics.items():
            df.loc[i, key] = value

    df_path = Path(exp_dir) / "experiments_results.csv"

    df.to_csv(
        df_path,
        index=False,
        encoding="utf-8-sig"
    )
    print(f"結果を {df_path} に保存しました。")
    return df


def make_enc_dec(args):
    args_combine = make_args_combine(args)
    
    log_file = args.log_dir + args_combine + ".log"
    model_file = args.model_dir + args_combine + ".pth"

    logger = setup_logging(log_file, filemode='a')
    
    print("Loading PointSet data..")

    pointset, feature_embedding = load_pointset(data_dir=args.data_dir, 
                                                    embed_dim=args.embed_dim,
                                                    do_feature_sampling = False)
    if args.cuda:
        pointset.feature_embed_lookup = cudify(feature_embedding)
        
    # make feature encoder
    enc = get_encoder(pointset.feature_embed_lookup, feature_embedding, pointset, args.enc_agg)

    if args.model_type == "relative" or args.model_type == "join" or args.model_type == "together":
        # make relative space encoder
        spa_enc = get_spa_encoder(args=args,
                            model_type=args.model_type,
                            spa_enc_type=args.spa_enc,
                            pointset=None, 
                            spa_embed_dim=args.spa_embed_dim, 
                            coord_dim = 2, 
                            frequency_num = args.freq, 
                            max_radius = args.max_radius,
                            # dropout = args.dropout,
                            freq_init = args.freq_init)
    else:
        spa_enc = None

    if args.model_type == "global" or args.model_type == "join" or args.model_type == "together":
        # make global space encoder
        g_spa_enc = get_spa_encoder(args=args,
                            model_type=args.model_type,
                            spa_enc_type=args.g_spa_enc, 
                            spa_embed_dim=args.g_spa_embed_dim,
                            pointset=None, 
                            coord_dim = 2, 
                            frequency_num = args.g_freq, 
                            max_radius = args.g_max_radius,
                            # dropout = args.dropout,
                            freq_init = args.g_freq_init)
    else:
        g_spa_enc = None

    # make decoder
    if args.model_type == "relative" or args.model_type == "join" or args.model_type == "together":

        # make query embedding initial decoder
        init_dec = get_context_decoder(dec_type=args.init_decoder_atten_type, 
                            query_dim=args.embed_dim, 
                            key_dim=args.embed_dim, 
                            spa_embed_dim=args.spa_embed_dim, 
                            g_spa_embed_dim=args.g_spa_embed_dim,
                            have_query_embed = False, 
                            num_attn = args.init_decoder_atten_num, 
                            activation = args.init_decoder_atten_act, 
                            f_activation = args.init_decoder_atten_f_act, 
                            layn = args.init_decoder_use_layn, 
                            use_postmat = args.init_decoder_use_postmat,
                            dropout = args.dropout)

        if args.use_dec == "T":
            # make decoder
            dec = get_context_decoder(dec_type=args.decoder_atten_type, 
                                query_dim=args.embed_dim, 
                                key_dim=args.embed_dim, 
                                spa_embed_dim=args.spa_embed_dim, 
                                g_spa_embed_dim=args.g_spa_embed_dim,
                                have_query_embed = True, 
                                num_attn = args.decoder_atten_num, 
                                activation = args.decoder_atten_act, 
                                f_activation = args.decoder_atten_f_act, 
                                layn = args.decoder_use_layn, 
                                use_postmat = args.decoder_use_postmat,
                                dropout = args.dropout)
        else:
            dec = None

        if args.model_type == "join":
            joint_dec = JointRelativeGlobalDecoder(feature_embed_dim = args.embed_dim, 
                            f_act = args.act, 
                            dropout = args.dropout,
                            join_type = args.join_dec_type)
        else:
            joint_dec = None

    else:
        init_dec = None
        dec = None
        joint_dec = None

    if args.model_type == "global" or args.model_type == "join":
        # make global space decoder
        g_spa_dec = DirectPositionEmbeddingDecoder(g_spa_embed_dim=args.g_spa_embed_dim, 
                            feature_embed_dim=args.embed_dim, 
                            f_act = args.act, 
                            dropout = args.dropout)
    else:
        g_spa_dec = None




    # if args.model_type == "global" or args.model_type == "relative":
    # make encoder encoder
    enc_dec = get_enc_dec(model_type=args.model_type, 
                        pointset=pointset, 
                        enc = enc, 
                        spa_enc = spa_enc, 
                        g_spa_enc = g_spa_enc, 
                        g_spa_dec = g_spa_dec, 
                        init_dec=init_dec, 
                        dec=dec, 
                        joint_dec=joint_dec, 
                        activation = args.act, 
                        num_context_sample = args.num_context_sample, 
                        num_neg_resample = 10)

    if args.cuda:
        enc_dec.cuda()

    if args.opt == "sgd":
        optimizer = optim.SGD(filter(lambda p : p.requires_grad, enc_dec.parameters()), lr=args.lr, momentum=0)
    elif args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p : p.requires_grad, enc_dec.parameters()), lr=args.lr)

#     logger.info("Save file at {}".format(args_combine + ".pth"))
    print("Load model from {}".format(args_combine + ".pth"))
    enc_dec.load_state_dict(torch.load(model_file))
    return enc_dec

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path


def plot_stg_survival_by_scale(
    module,
    threshold=0.5,
    save_dir=None,
):
    assert module.stg is not None, "STG is not enabled"

    # STGゲートの値（評価時）
    gate = module.stg.mu.detach().cpu().clamp(0, 1).numpy()

    # 基底点とスケール
    bases = module.bases.detach().cpu().numpy()
    scales = module.scales.detach().cpu().numpy()

    unique_scales = np.unique(scales)

    # 各基底点に対応するSTGゲートを復元
    # j=0: 4種類、j>=1: 3種類
    n0 = module.n_level0

    gate_level0 = gate[:n0 * 4].reshape(n0, 4)
    gate_other = gate[n0 * 4:].reshape(-1, 3)

    # 各基底点で生き残った基底の割合
    survival = np.concatenate([
        (gate_level0 > threshold).mean(axis=1),
        (gate_other > threshold).mean(axis=1)
    ])

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    xmin, xmax, ymin, ymax = module.extent

    for scale in unique_scales:
        mask = scales == scale

        xy = bases[mask]
        values = survival[mask]

        fig, ax = plt.subplots(figsize=(8, 10))

        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=values,
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=80,
            marker="s",
            edgecolors="none"
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")

        ax.set_title(
            f"Scale = {scale:g} m\n"
            f"Survival rate = {values.mean():.3f}"
        )

        fig.colorbar(
            sc,
            ax=ax,
            label="Fraction of active wavelet bases"
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        if save_dir is not None:
            fig.savefig(
                save_dir / f"stg_scale_{scale:g}.png",
                dpi=150,
                bbox_inches="tight"
            )

        plt.show()
        plt.close(fig)

def _extract_stg_coeff_by_scale(module):
    """
    STGの mu を基底タイプごとに復元する。
    返り値:
        result[scale] = {
            "bases": (n_basis, 2),
            "phph": (n_basis,),   # j=0 以外は None
            "phps": (n_basis,),
            "psph": (n_basis,),
            "psps": (n_basis,),
        }
    """
    assert module.stg is not None, "module.stg is None"
    
    gate = module.stg.mu.detach().cpu().clamp(0, 1).numpy()
    bases = module.bases.detach().cpu().numpy()
    scales = module.scales.detach().cpu().numpy()

    n0 = module.n_level0

    # j=0 部分
    # make_input_embeds 内の
    # level0 = cat([phph.unsqueeze(-1), level0], dim=-1)
    # より、順序は [φφ, (元の3種類...)] になっている
    level0_gate = gate[:n0 * 4].reshape(n0, 4)

    # ここで「元の3種類」の順序は wlfunc の実装に依存
    # 通常ここでは [φψ, ψφ, ψψ] と仮定する
    # もし違えばここを入れ替えてください
    phph0 = level0_gate[:, 0]
    phps0 = level0_gate[:, 1]
    psph0 = level0_gate[:, 2]
    psps0 = level0_gate[:, 3]

    result = {}

    unique_scales = np.unique(scales)
    min_scale = unique_scales.min()

    # j=0
    mask0 = scales == min_scale
    bases0 = bases[mask0]
    result[min_scale] = {
        "bases": bases0,
        "phph": phph0,
        "phps": phps0,
        "psph": psph0,
        "psps": psps0,
    }

    # j>=1
    other_scales = unique_scales[unique_scales != min_scale]
    other_gate = gate[n0 * 4:].reshape(-1, 3)

    start = 0
    for sc in other_scales:
        mask = scales == sc
        cnt = mask.sum()
        part = other_gate[start:start + cnt]
        start += cnt

        result[sc] = {
            "bases": bases[mask],
            "phph": None,          # j>=1 には存在しない
            "phps": part[:, 0],
            "psph": part[:, 1],
            "psps": part[:, 2],
        }

    return result


def plot_stg_basis_type_map(
    module,
    save_dir=None,
    show=True,
    dpi=150,
    img_path="map_cache/las_vegas.png",
    img_extent_path="map_cache/las_vegas_extent.json"
):
    data = _extract_stg_coeff_by_scale(module)

    if img_path is not None:
        img_path = Path(img_path)
        if not img_path.is_file():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        img = Image.open(img_path)

        with open(img_extent_path) as f:
            img_extent = json.load(f)["extent"]

    xmin, xmax, ymin, ymax = module.extent

    colors = {
        "phph": "#d62728",  # 赤
        "phps": "#1f77b4",  # 青
        "psph": "#2ca02c",  # 緑
        "psps": "#ff7f0e",  # オレンジ
    }

    # 各区画の相対位置
    offsets = {
        "phph": (0, 1),  # 左上
        "phps": (1, 1),  # 右上
        "psph": (0, 0),  # 左下
        "psps": (1, 0),  # 右下
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for scale, d in sorted(data.items()):

        bases = d["bases"]
        half = scale / 2

        fig, ax = plt.subplots(figsize=(9, 12))

        if img_path is not None:
            ax.imshow(
                img,
                extent=img_extent,
                origin="upper",
                alpha=0.5,
                zorder=0
            )

        patches = []
        facecolors = []

        # 各基底点に一辺scaleの正方形を配置
        for i, (x, y) in enumerate(bases):

            for key in ["phph", "phps", "psph", "psps"]:

                dx, dy = offsets[key]

                rect = Rectangle(
                    (x + dx * half, y + dy * half),
                    half,
                    half
                )

                patches.append(rect)

                vals = d[key]

                if vals is None:
                    rgba = (0, 0, 0, 0)
                else:
                    alpha = float(vals[i])
                    rgb = plt.matplotlib.colors.to_rgb(colors[key])
                    rgba = (*rgb, alpha)

                facecolors.append(rgba)

        # 4分割した区画
        collection = PatchCollection(
            patches,
            facecolor=facecolors,
            edgecolor="none",
            zorder=2
        )
        ax.add_collection(collection)

        # 正方形全体の外枠
        outer_patches = []

        for i, (x, y) in enumerate(bases):
            active = any(
                d[key] is not None and d[key][i] > 0
                for key in ["phph", "phps", "psph", "psps"]
            )

            if active:
                outer_patches.append(
                    Rectangle((x, y), scale, scale)
                )

        outer_collection = PatchCollection(
            outer_patches,
            facecolor="none",
            edgecolor="black",
            linewidth=0.35,
            alpha=0.5,
            zorder=3
        )
        ax.add_collection(outer_collection)

        # 平均ゲート値と非ゼロ率
        all_values = np.concatenate([
            d[key]
            for key in ["phph", "phps", "psph", "psps"]
            if d[key] is not None
        ])

        mean_gate = all_values.mean()
        active_rate = (all_values > 0).mean()

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")

        ax.set_title(
            f"STG basis map | Scale = {scale:g} m\n"
            f"Mean gate = {mean_gate:.3f}, "
            f"Active rate = {active_rate:.3%}"
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        legend_handles = [
            Patch(facecolor=c, edgecolor="black", label=label)
            for label, c in zip(
                ["φφ", "φψ", "ψφ", "ψψ"],
                colors.values()
            )
        ]

        ax.legend(
            handles=legend_handles,
            loc="upper right"
        )

        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(
                save_dir / f"stg_scale_{scale:g}.png",
                dpi=dpi,
                bbox_inches="tight"
            )

        if show:
            plt.show()

        plt.close(fig)

        print(
            f"Scale {scale:g} m | "
            f"Basis points: {len(bases)} | "
            f"Mean gate: {mean_gate:.4f} | "
            f"Active: {active_rate:.2%}"
        )