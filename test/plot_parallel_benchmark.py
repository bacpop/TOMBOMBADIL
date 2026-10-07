"""Render cold-fit objective histories from a PERF-05 JSON archive."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive");parser.add_argument("--output",required=True)
    args=parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    reports=json.loads(Path(args.archive).read_text())["reports"]
    fig,axes=plt.subplots(3,1,figsize=(11,10),constrained_layout=True)
    colors={"1:5":"#d97706","5:20":"#7c3aed"}
    for start,ax in enumerate(axes):
        base=next(r for r in reports if r["report_file"]==f"baseline-map-start{start}-1.json")
        target=base["timed"]["objective"]
        tolerance=1e-8+1e-6*abs(target)
        def draw(history,label,color,style="-"):
            ax.plot([h["seconds"] for h in history],[target-h["objective"] for h in history],
                    label=label,color=color,linestyle=style,linewidth=1.6)
        draw(base["warmup"]["history"],"Joint Adam (cold fit)","#0284c7")
        for r in reports:
            a=r["arguments"]
            if a["mode"]!="alternate" or a["start"]!=start or a["site_repeat"]!=1:
                continue
            if not r["report_file"].endswith("-1.json"): continue
            parallel=a["variant"]!="baseline"
            label=f"{'Parallel omega' if parallel else 'Serial'} {a['rounds']}"
            complete="alternating" in r
            history=r["alternating"]["history"] if complete else r.get("partial_history",[])
            if not complete: label+=" (incomplete)"
            draw(history,label,"#16a34a" if parallel else colors[a["rounds"]],"--" if parallel else "-")
        ax.axhspan(-tolerance,tolerance,color="#cbd5e1",alpha=.4,label="Objective tolerance band")
        ax.set_yscale("symlog",linthresh=tolerance)
        ax.set_title("Standard start" if start==0 else f"Perturbed start {start}",loc="left")
        ax.set_ylabel("Baseline endpoint − log density\n(lower is better)")
        ax.set_xlabel("Fit wall time (seconds)")
        ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=3,loc="upper right")
    fig.suptitle("PERF-05: joint and alternating optimisation\nFirst trials; compilation and cycle evaluations included",fontsize=14)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=150)
    plt.close(fig)


if __name__=="__main__":
    main()
