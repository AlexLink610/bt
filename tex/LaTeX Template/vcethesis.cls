% ===============================================
% ============= Acknowledgement =================
% ===============================================
% This template is an Adaptation of Tobias Günter's
% template.
% It was adapted by Mathias Harrer (mathias.mh.harer@fau.de)
% for the use at Visual Computing Erlangen.
% ===============================================
% ================== Class ======================
% ===============================================
% export a new document class
% (based on the standard "report" class)
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{vcethesis}[2025/02/11]
\DeclareOption*{\PassOptionsToClass{\CurrentOption, 11pt, twoside, openright, a4paper}{report}}
\ProcessOptions

\LoadClass{report}

% ===============================================
% ================= Packages ====================
% ===============================================
\AtEndOfPackage{
\addbibresource{bibliography.bib}
\RequirePackage{graphicx,import}
\RequirePackage{color}
\RequirePackage{booktabs}
\RequirePackage{siunitx}
\RequirePackage[mono=false]{libertine}
\RequirePackage{libertinust1math}
\RequirePackage[T1]{fontenc}
\RequirePackage{multirow}
\RequirePackage{algorithmic}
\RequirePackage[boxed,chapter]{algorithm}
\RequirePackage{subcaption}
\RequirePackage[margin=10pt,font=small,labelfont=bf]{caption}
\RequirePackage{microtype}
\RequirePackage{tikz}
\usetikzlibrary{shapes,arrows}
\usetikzlibrary{positioning}
\usetikzlibrary{calc}
\RequirePackage[hidelinks]{hyperref}

}

\RequirePackage[utf8]{inputenc}
\RequirePackage[backend=biber,style=numeric,bibencoding=utf8,natbib=true, maxbibnames=20]{biblatex}
\RequirePackage{amsfonts}	
\RequirePackage{amsmath}
\RequirePackage{amssymb}

% ===============================================
% ================ Page Setup ===================
% ===============================================
% set the margins for printing twoside:
%   margin at the center: 
%       -even pages: larger margin on the right
%       -odd pages:  larger margin on the left
\newlength{\ML}
\setlength{\ML}{\dimexpr 1in-3mm\relax}
% twoside and margins
\RequirePackage[a4paper, top=1.6in, bottom=1.284in, headheight=13.6pt, headsep=20pt,
    hoffset=0pt, twoside, lmargin=\ML, rmargin=\ML, bindingoffset=6mm]{geometry}
% Marginal notes:
\setlength{\marginparwidth}{0.90in}     % Width of marginal notes.
\setlength{\marginparsep}{0.10in}       % space between outer margin and
                                        % marginal note
                                        
\renewcommand{\topfraction}{0.999}
\renewcommand{\textfraction}{0.001}
\setcounter{topnumber}{10}
\setcounter{dbltopnumber}{10}
\setcounter{totalnumber}{10}
\renewcommand{\dbltopfraction}{0.999}
\renewcommand{\dblfloatpagefraction}{0.999}

% increase line spacing to make things more readable
\renewcommand{\baselinestretch}{1.1}

\RequirePackage{fancyhdr}
\rhead[]{\fancyplain{}{\rightmark}}
\lhead[{\fancyplain{}{\leftmark}}]{}
\pagestyle{fancyplain}

\frenchspacing

% numbering schemes
\def\prepages{\normalfont\pagenumbering{roman}}
\def\mainbody{\pagenumbering{arabic}}

% ===============================================
% ============ Listing Style Setup ==============
% ===============================================
\RequirePackage{listings}
\renewcommand{\lstlistlistingname}{List of Listings}
\RequirePackage[table]{xcolor}
\definecolor{codecolor}{rgb}{0.15,0.15,0.15}
\definecolor{commentcolor}{rgb}{0.4,0.4,0.4}
\definecolor{codegray}{rgb}{0.15,0.15,0.15}
\definecolor{keywordcolor}{rgb}{0,0,1}
\definecolor{backcolour}{rgb}{1,1,1}
\definecolor{stringcolor}{rgb}{0,0.6,0.6}
\definecolor{functioncolor}{rgb}{0.6,0,0}
\definecolor{typecolor}{rgb}{0,0,0.5}
\definecolor{namespacecolor}{rgb}{0,0.5,0}

\lstdefinestyle{mystyle}
{
	basicstyle=\small\ttfamily\color{codecolor},
	frame=single,
	backgroundcolor=\color{backcolour},   
	commentstyle=\color{commentcolor},
	keywordstyle=\color{keywordcolor},
	numberstyle=\tiny\color{codegray},
	stringstyle=\color{stringcolor},
	breakatwhitespace=true,         
	breaklines=true,                 
	captionpos=b,                    
	keepspaces=true,                 
	numbers=left,                    
	showspaces=false,                
	showstringspaces=false,
	showtabs=false,                  
	tabsize=2
}
\lstset{style=mystyle}

% ===============================================
% ============== Title Page Data ================
% ===============================================
% title page macros
\newcommand\thesistype[1]{\gdef\@thesistype{#1}}
\newcommand\birthday[1]{\gdef\@birthday{#1}}
\newcommand\birthplace[1]{\gdef\@birthplace{#1}}
\newcommand\supervisor[1]{\gdef\@supervisor{Supervisor:~ #1\\}}
\newcommand\supervisors[1]{\gdef\@supervisor{Supervisors:~ #1\\}}
\newcommand\mentor[1]{\gdef\@mentor{#1}}
\newcommand\bdate[1]{\gdef\@bdate{#1}}
\newcommand\edate[1]{\gdef\@edate{#1}}
\newcommand\institute[1]{\gdef\@institute{#1}}
\newcommand\additionallogos[1]{\gdef\@additionallogos{#1}}
\newcommand\aideclaration[1]{\gdef\@aideclaration{#1}}

\newcommand{\logofooter}{%
\noindent%
\smash{\raisebox{-12mm}{%
\hspace*{-0.05\textwidth}%
\begin{minipage}{1.1\textwidth}
\newlength{\logoheight}
\setlength{\logoheight}{0.05\linewidth}
\includegraphics[height=\logoheight]{logos/vce/black-stencil-right-label-tight.pdf}%
\hfill%
\@additionallogos%
\institutelogo%
\end{minipage}
\hspace*{-0.05\textwidth}%
\strut\\}}}

\gdef\@thesistype{$<$TYPE OF THESIS$>$}
\gdef\@title{$<$TITLE$>$}
\gdef\@author{$<$AUTHOR$>$}
\gdef\@birthday{$<$BIRTHDAY$>$}
\gdef\@birthplace{$<$BIRTHPLACE$>$}
\gdef\@supervisor{}
\gdef\@mentor{$<$MENTOR$>$}
\gdef\@bdate{$<$BEGINNING OF THESIS$>$}
\gdef\@edate{$<$END OF THESIS$>$}
\gdef\@institute{$<$INSTITUTE$>$}
\gdef\@additionallogos{}
\gdef\@aideclaration{$<$AI DECLARATION$>$}

% ===============================================
% =================== Logos =====================
% ===============================================
% logos for footer of titlepage
\newcommand{\logoweyrich}{\quad\includegraphics[height=\logoheight]{logos/dr/black-on-white-no-label-tight.pdf}}
\newcommand{\logostamminger}{\quad\includegraphics[height=1.22162\logoheight]{logos/lgdv/lgdv_logo_cg.pdf}}
%\newcommand{\logoguenther}{\quad\includegraphics[height=1.22162\logoheight]{logos/lgdv/lgdv_logo_vc.pdf}}
\newcommand{\logoegger}{\quad\includegraphics[height=2.25\logoheight]{logos/cogcovi/cogcovi.jpg}}
% ===============================================
% =========== Institute - english ===============
% ===============================================
\newcommand{\institutelogo}{}
\mentor{$<$ACADEMIC SUPERVISION$>$}
\institute{$<$INSTITUTE$>$}
% Weyrich
\DeclareOption{weyrich}{
    \def\academicSupervision{weyrich}
    \renewcommand{\institutelogo}{\logoweyrich}
    \mentor{Prof. Dr. Tim Weyrich}
    \institute{Chair of Digital Reality}
}
% Stamminger
\DeclareOption{stamminger}{
    \def\academicSupervision{stamminger}
    \renewcommand{\institutelogo}{\logostamminger}
    \mentor{Prof. Dr.-Ing. Marc Stamminger}
    \institute{Chair of Computer Graphics}
}

% Guenther
\DeclareOption{guenther}{
    \def\academicSupervision{guenther}
    %\renewcommand{\institutelogo}{\logoguenther}
    \renewcommand{\institutelogo}{}
    \mentor{Prof. Dr.-Ing. Tobias G{\"u}nther}
    \institute{Professorship for Visual Computing}
}

% Egger
\DeclareOption{egger}{
    \def\academicSupervision{egger}
    \renewcommand{\institutelogo}{\logoegger}
    \mentor{Prof. Dr. Bernhard Egger}
    \institute{Professorship for Cognitive Computer Vision}
    
    % \AtEndOfPackage{
    % \usepackage{fontspec}
    % \setmainfont{Comic Sans MS}
    % \RequirePackage{lettrine}
    % \usepackage[lines=5]{lettrine}
    % \usepackage{GoudyIn}
    % \usepackage[x11names]{xcolor} 
    % \renewcommand{\LettrineFontHook}{\color{VioletRed4}\GoudyInfamily{}}
    % \LettrineTextFont{\itshape}
    % }
}
% ===============================================
% ========== References - english ===============
% ===============================================
\renewcommand{\eqref}[1]{Equation~(\ref{#1})}
\newcommand{\figref}[1]{Figure~\ref{#1}}
\newcommand{\secref}[1]{Section~\ref{#1}}
\newcommand{\tblref}[1]{Table~\ref{#1}}
\newcommand{\lstref}[1]{Listing~\ref{#1}}
% ===============================================
% ========== Thesis Type - english ==============
% ===============================================
\DeclareOption{bsc}{\thesistype{Bachelor Thesis}}
\DeclareOption{msc}{\thesistype{Master Thesis}}

% ===============================================
% ========== Title Page - english ===============
% ===============================================
\renewcommand{\maketitle}{%
\begin{titlepage} \setlength{\parindent}{0pt}
\thispagestyle{empty}
\begin{center}
{\LARGE\bf\@title}
\vspace{1.50cm}

{\large \@thesistype\ in Computer Science}
\vspace{1.50cm}

{\small presented by}\bigskip

{\bf\@author}\bigskip

born\ on \@birthday\ in \@birthplace\bigskip
\vspace{1.5cm}

{\small submitted to}\bigskip

{\bf Department of Computer Science\\
\@institute\\
Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg}
\vfill

\@supervisor
\bigskip
Academic Supervision:~ \@mentor
\vfill

Begin date:\ \@bdate\bigskip

End date:\ \@edate
\end{center}

\vfill
\vfill
\logofooter
\newpage
\end{titlepage}
}

% ===============================================
% ========= Declaration - english ===============
% ===============================================
\newcommand{\declaration}{%
\parindent0pt
\chapter*{Declaration}
\bigskip

I certify that I have written this thesis without outside help and without using sources other than those indicated, and that the thesis has not been submitted in the same or similar form to any other examination authority and has not been accepted by them as part of an examination. All statements that have been taken over verbatim or in spirit are marked as such.

\bigskip
\@aideclaration

\bigskip
I agree that the thesis may be published and that it may be referred to in scientific publications.

\bigskip
The Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, represented by the \@institute, is granted a (non-exclusive) right to use this work and the programs created in connection with it.

\vskip3cm
Erlangen, \@edate \hfill

\hfill{\vbox{\hrule{\hbox{\strut$\qquad$\small(\@author)$\qquad$}}}}
\thispagestyle{empty}
}

% ===============================================
% ============ Packages - german ================
% ===============================================
\DeclareOption{german}{
% enable German chapter captions
\AtEndOfPackage{
\RequirePackage{csquotes}
\RequirePackage[ngerman]{babel}    
% use et al. rather than u. a.
\DefineBibliographyStrings{ngerman}{
   andothers = {{et\,al\adddot}},
}
}
% ===============================================
% ========== Thesis Type - german ===============
% ===============================================
\DeclareOption{bsc}{\thesistype{Bachelorarbeit}}
\DeclareOption{msc}{\thesistype{Masterarbeit}}

% ===============================================
% ============ Institute - german ===============
% ===============================================
% Weyrich
\newcommand\pronoun[1]{\gdef\@pronoun{#1}}
\gdef\@pronoun{}
\pronoun{den }
\DeclareOption{weyrich}{
    \institute{Lehrstuhl für Digital Reality}
}
% Stamminger
\DeclareOption{stamminger}{
    \institute{Lehrstuhl für Graphische Datenverarbeitung}
}

% Guenther
\DeclareOption{guenther}{
    \institute{Professur für Visual Computing}
    \pronoun{die }
}

% Egger
\DeclareOption{egger}{
    \institute{Professur für Cognitive Computer Vision}
    \pronoun{die }
}

% ===============================================
% ========== References - german ===============
% ===============================================
\renewcommand{\eqref}[1]{Gleichung~(\ref{#1})}
\renewcommand{\figref}[1]{Abbildung~\ref{#1}}
\renewcommand{\secref}[1]{Abschnitt~\ref{#1}}
\renewcommand{\tblref}[1]{Tabelle~\ref{#1}}
\renewcommand{\lstref}[1]{Auflistung~\ref{#1}}
\renewcommand{\lstlistingname}{Auflistung}
\renewcommand{\lstlistlistingname}{Auflistungsverzeichnis}
% ===============================================
% ========== Title Page - german ================
% ===============================================
\newcommand\supervisorM[1]{\gdef\@supervisor{Betreuer:~ #1\\}}
\newcommand\supervisorF[1]{\gdef\@supervisor{Betreuerin:~ #1\\}}
\newcommand\supervisorD[1]{\gdef\@supervisor{Betreuende Person:~ #1\\}}
\renewcommand\supervisors[1]{\gdef\@supervisor{Betreuende:~ #1\\}}
\renewcommand\supervisor[1]{\supervisorM{#1}}

\renewcommand{\maketitle}{%
\begin{titlepage} \setlength{\parindent}{0pt}
\thispagestyle{empty}
\begin{center}
{\LARGE\bf\@title}
\vspace{1.50cm}

{\large \@thesistype\ in Informatik}
\vspace{1.50cm}

{\small vorgelegt von}\bigskip

{\bf\@author}\bigskip

geb.\ am \@birthday\ in \@birthplace\bigskip
\vspace{1.5cm}

{\small angefertig am}\bigskip

{\bf Department Informatik\\
\@institute\\
Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg}
\vfill

\@supervisor
\bigskip
Akademische Betreuung:~ \@mentor
\vfill
\vfill

Beginn der Arbeit:\ \@bdate\bigskip

Ende der Arbeit:\ \@edate
\vfill

\end{center}

\vfill
\vfill
\logofooter
\newpage
\end{titlepage}
}

% ===============================================
% ========= Declaration - german ================
% ===============================================
\renewcommand{\declaration}{%
\parindent0pt
\chapter*{Erkl\"arung}
\bigskip

Ich versichere, dass ich die Arbeit ohne fremde Hilfe und ohne Benutzung
anderer als der angegebenen Quellen angefertigt habe und dass die Arbeit in
gleicher oder \"ahnlicher Form noch keiner anderen Pr\"ufungsbeh\"orde
vorgelegen hat und von dieser als Teil einer Pr\"ufungsleistung angenommen
wurde. Alle Ausf\"uhrungen, die w\"ortlich oder sinngem\"a\ss\ \"ubernommen
wurden, sind als solche gekennzeichnet.

\bigskip
\@aideclaration

\bigskip
Ich bin damit einverstanden, dass die Arbeit ver\"offentlicht wird und
dass in wissenschaftlichen Ver\-\"offent\-lich\-ungen auf sie Bezug genommen wird.

\bigskip
Der Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, vertreten durch \@pronoun \@institute, wird ein (nicht ausschlie\ss liches)
Nutzungsrecht an dieser Arbeit sowie an den im Zusammenhang mit ihr
erstellten Programmen einger\"aumt.

\vskip3cm
Erlangen, den \@edate \hfill

\hfill{\vbox{\hrule{\hbox{\strut$\qquad$\small(\@author)$\qquad$}}}}
\thispagestyle{empty}
}
}
\ProcessOptions\relax