import subprocess
TPL = r"""\documentclass[tikz,border=4pt]{standalone}
\usepackage[scaled]{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usetikzlibrary{arrows.meta}
\definecolor{bfill}{HTML}{ECECFF}
\definecolor{bdraw}{HTML}{9C7CDE}
\definecolor{dfill}{HTML}{F2F2F2}
\begin{document}
\begin{tikzpicture}[
  box/.style={draw=bdraw, fill=bfill, line width=0.8pt,
    rounded corners=2pt, align=center, minimum width=2.6cm,
    minimum height=1.1cm, font=\small},
  data/.style={box, draw=black!60, fill=dfill},
  flow/.style={-{Stealth[length=2mm]}, thick},
  loop/.style={-{Stealth[length=2mm]}, thick, dashed}]
\node[data, opacity=%(img)s] (img) at (0,0) {Input images};
\node[box, opacity=%(vggt)s] (vggt) at (3.1,0.8) {VGGT\\point maps};
\node[box, opacity=%(sam)s] (sam) at (3.1,-0.8) {SAM3\\instance masks};
\node[box, opacity=%(filt)s] (filt) at (6.2,0) {Outlier\\removal};
\node[box, opacity=%(match)s] (match) at (9.3,0) {Pairwise\\matching};
\node[box, opacity=%(agg)s] (agg) at (9.3,-3) {Aggregation};
\node[box, opacity=%(apple)s] (apple) at (6.2,-3) {Apple model\\(sphere fit)};
\node[box, opacity=%(clus)s] (clus) at (3.1,-3) {Clustering};
\node[data, opacity=%(count)s] (count) at (0,-3) {Count};
\draw[flow, opacity=%(e1)s] (img) -- (vggt);
\draw[flow, opacity=%(e1)s] (img) -- (sam);
\draw[flow, opacity=%(e2)s] (vggt) -- (filt);
\draw[flow, opacity=%(e2)s] (sam) -- (filt);
\draw[flow, opacity=%(e3)s] (filt) -- (match);
\draw[flow, opacity=%(e4)s] (match) -- (agg);
\draw[flow, opacity=%(e5)s] (agg) -- (apple);
\draw[flow, opacity=%(e6)s] (apple) -- (clus);
\draw[flow, opacity=%(e6)s] (clus) -- (count);
\draw[loop, opacity=%(loop)s] (apple.north) --
  node[left, font=\footnotesize, align=right] {refinement}
  (match.south west);
\end{tikzpicture}
\end{document}
"""
keys = "img vggt sam filt match agg apple clus count e1 e2 e3 e4 e5 e6 loop".split()
F = 0.18
variants = {
 "overview":    set(keys),
 "input":       {"img","vggt","sam","filt","e1","e2"},
 "association": {"match","agg","e3","e4"},
 "counting":    {"apple","clus","count","e5","e6"},
 "refinement":  {"match","agg","apple","e4","e5","loop"},
}
for name, on in variants.items():
    d = {k: (1 if k in on else F) for k in keys}
    fn = f"pipeline_{name}"
    open(fn+".tex","w").write(TPL % d)
    subprocess.run(["pdflatex","-interaction=batchmode",fn+".tex"],check=True)
    subprocess.run(["pdftocairo","-png","-transp","-r","300","-singlefile",fn+".pdf",fn],check=True)
    subprocess.run(["pdftocairo","-svg",fn+".pdf",fn+".svg"],check=True)
