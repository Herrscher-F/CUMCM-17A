\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{MM_thesis}
              [2025/05/27 v1.0 Standard LaTeX Template for Mathematical Modeling]

%% =========================================================================
%% 第一部分：选项声明与基础设置
%% =========================================================================

% 初始化标记
\newif\if@mcm@bwprint\@mcm@bwprintfalse

% 令牌定义
\newcommand\mcm@tokens@keywords{}

% 选项声明
\DeclareOption{colorprint}{\@mcm@bwprintfalse}
\DeclareOption{bwprint}{\@mcm@bwprinttrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

% 执行选项
\ExecuteOptions{colorprint}
\ProcessOptions\relax

% 加载基础文档类
\LoadClass[a4paper,12pt]{article}

%% =========================================================================
%% 第二部分：宏包依赖管理
%% =========================================================================

%% 2.1 核心依赖与编译检查
\RequirePackage{ifxetex}
\RequireXeTeX
\ifxetex\else
\ClassError{mcmthesis}{You must use the `xelatex' driver\MessageBreak Please choose `xelatex'}{%
Just choose `xelatex', no `pdflatex' or `latex' and so on.}
\fi

%% 2.2 中文支持与字体管理
\RequirePackage{ctex}              % 中文支持
\RequirePackage{fontspec}          % 字体管理
\RequirePackage{unicode-math}      % 现代数学字体

%% 2.3 页面布局与间距
\RequirePackage{geometry}          % 页面布局
\RequirePackage{setspace}          % 行间距控制
\RequirePackage{calc}              % 尺寸计算

%% 2.4 颜色与图形
\RequirePackage{xcolor}            % 颜色设置
\RequirePackage{graphicx}          % 图片插入
\RequirePackage{float}             % 浮动环境控制
\RequirePackage{placeins}          % 浮动环境约束

%% 2.5 表格与列表
\RequirePackage{tabularray}        % 现代表格解决方案
\RequirePackage{booktabs}          % 表格线条美化
\RequirePackage{enumitem}          % 列表环境定制

%% 2.6 标题与目录
\RequirePackage{titlesec}          % 标题格式设置
\RequirePackage[titles]{tocloft}   % 目录格式定制
\RequirePackage{caption}           % 图表标题格式
\RequirePackage{subcaption}        % 子图表标题

%% 2.7 代码高亮
\RequirePackage{minted}            % 代码高亮与语法着色
\RequirePackage{xcolor}            % 颜色定义

%% 2.8 数学环境
\RequirePackage{amsthm}            % 数学定理环境

%% 2.9 参考文献与超链接
\RequirePackage[
    backend=biber,
    style=gb7714-2015,
    sorting=none,
    giveninits=true,
    backref=false,
    natbib=true
]{biblatex}
\addbibresource{myrefs.bib}

\RequirePackage{hyperref}          % 超链接支持
\RequirePackage{cleveref}          % 智能交叉引用

%% 2.10 其他工具宏包
\RequirePackage{url}               % URL 处理
\RequirePackage{ulem}              % 下划线
\RequirePackage[titletoc,title]{appendix}  % 附录
\RequirePackage{etoolbox}          % 工具宏

%% =========================================================================
%% 第三部分：页面布局与间距设置
%% =========================================================================

% 页面布局
\geometry{top=25.4mm,bottom=25.4mm,left=27.0mm,right=27.0mm}

% 行间距设置
\renewcommand*{\baselinestretch}{1.35}

% 段落缩进
\setlength\parindent{2em}

% 表格行间距修正
\let\mcm@oldtabular\tabular
\let\mcm@endoldtabular\endtabular
\renewenvironment{tabular}%
	{\bgroup%
	\renewcommand{\arraystretch}{1.35}%
	\mcm@oldtabular}%
	{\mcm@endoldtabular\egroup}

%% =========================================================================
%% 第四部分：字体配置
%% =========================================================================

%% 4.1 西文字体设置
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Consolas}[Scale=0.9]

%% 4.2 数学字体设置
\setmathfont{XITS Math}

%% 4.3 中文字体设置
\setCJKmainfont{SimSun}[AutoFakeBold=2]
\setCJKsansfont{Microsoft YaHei}[AutoFakeBold=2]
\setCJKmonofont{FangSong}

%% 4.4 特殊字体定义
\setCJKfamilyfont{kai}{KaiTi}[AutoFakeBold=2]
\newcommand*{\kai}{\CJKfamily{kai}}
\setCJKfamilyfont{song}{SimSun}[AutoFakeBold=2]
\newcommand*{\song}{\CJKfamily{song}}

%% 4.5 字号设置
\renewcommand\normalsize{%
	\@setfontsize\normalsize{12.05}{14.45}%
	\abovedisplayskip 12\p@ \@plus3\p@ \@minus7\p@
	\abovedisplayshortskip \z@ \@plus3\p@
	\belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@}

%% =========================================================================
%% 第五部分：标题与章节格式
%% =========================================================================

%% 5.1 章节标题格式
\titleformat{\section}{\centering\Large\bfseries}{\chinese{section}、}{0pt}{}
\titlespacing{\section}{0pt}{3.5ex plus 1ex minus 0.2ex}{2.3ex plus 0.2ex}

%% 5.2 目录格式设置
\renewcommand{\cftsecdotsep}{4.5}
\setlength{\cftbeforesecskip}{7pt}
\setlength{\cftbeforesubsecskip}{3pt}
\renewcommand{\cftsecfont}{\bfseries\zihao{-4}\heiti}
\renewcommand{\cftsecleader}{\cftdotfill{\cftsecdotsep}}
\renewcommand{\cftsecaftersnumb}{\hskip.4em}

%% =========================================================================
%% 第六部分：数学环境定义
%% =========================================================================

%% 6.1 定理类环境定义
\theoremstyle{definition}
\newtheorem{definition}{\mcm@cap@definition}
\newtheorem{theorem}{\mcm@cap@theorem}
\newtheorem{lemma}{\mcm@cap@lemma}
\newtheorem{corollary}{\mcm@cap@corollary}
\newtheorem{assumption}{\mcm@cap@assumption}
\newtheorem{conjecture}{\mcm@cap@conjecture}
\newtheorem{axiom}{\mcm@cap@axiom}
\newtheorem{principle}{\mcm@cap@principle}
\newtheorem{problem}{\mcm@cap@problem}
\newtheorem{example}{\mcm@cap@example}

%% 6.2 证明环境
\renewcommand{\proof}{\noindent\textbf{\mcm@cap@proof}\quad}
\newtheorem{solution}{\mcm@cap@solution}

%% =========================================================================
%% 第七部分：代码环境配置
%% =========================================================================

%% 7.1 minted 全局设置
\setminted{
    fontsize=\small,
    linenos=false,
    frame=single,
    framesep=2mm,
    bgcolor=white,
    breaklines=true,
    breakanywhere=true,
    tabsize=4,
    xleftmargin=0pt,
    xrightmargin=0pt,
    numbersep=8pt
}

%% 7.2 自定义代码样式
\newminted{python}{
    fontsize=\small,
    linenos=true,
    frame=single,
    framerule=1pt,
    framesep=10pt,
    bgcolor=gray!5,
    breaklines=true,
    tabsize=4
}

% \newminted{matlab}{
%     fontsize=\small,
%     linenos=true,
%     frame=leftline,
%     framerule=1pt,
%     framesep=10pt,
%     bgcolor=blue!5,
%     breaklines=true,
%     tabsize=4
% }

% \newminted{cpp}{
%     fontsize=\small,
%     linenos=true,
%     frame=leftline,
%     framerule=1pt,
%     framesep=10pt,
%     bgcolor=green!5,
%     breaklines=true,
%     tabsize=4
% }

% %% 7.3 颜色主题定义（如需自定义样式）
% \definecolor{codebg}{rgb}{0.95,0.95,0.95}
% \definecolor{codeframe}{rgb}{0.8,0.8,0.8}

%% =========================================================================
%% 第八部分：图表配置
%% =========================================================================

%% 8.1 浮动环境设置
\renewcommand*{\textfraction}{0.05}
\renewcommand*{\topfraction}{0.9}
\renewcommand*{\bottomfraction}{0.8}
\renewcommand*{\floatpagefraction}{0.85}

%% 8.2 图片路径与格式
\DeclareGraphicsExtensions{.pdf,.eps,.jpg,.png}
\graphicspath{{figs/}{figures/}{figure/}{pictures/}%
  {picture/}{pic/}{pics/}{image/}{images/}}

%% 8.3 图表标题格式
\captionsetup{
    font={small,bf},
    labelsep=quad,
    format=hang
}
\captionsetup[figure]{position=bottom}
\captionsetup[table]{position=top}

%% =========================================================================
%% 第九部分：列表环境配置
%% =========================================================================

\setlist{
	topsep=0.3em,
	partopsep=0pt,
	itemsep=0ex plus 0.1ex,
	parsep=0pt,
	leftmargin=1.5em,
	rightmargin=0em,
	labelsep=0.5em,
	labelwidth=2em
}

%% =========================================================================
%% 第十部分：参考文献配置
%% =========================================================================

%% 10.1 biblatex 中文设置
\DefineBibliographyStrings{english}{%
  bibliography = {参考文献},
}

%% 10.2 参考文献格式
\renewcommand*{\bibfont}{\small}
\setlength{\bibitemsep}{0.5ex}
\setlength{\bibhang}{2em}

\defbibheading{bibliography}[\bibname]{%
  \section*{#1}%
  \markboth{#1}{#1}
  \phantomsection
  \addcontentsline{toc}{section}{#1}
}

% %% 10.3 引用命令别名
% \newcommand{\upcite}[1]{\textsuperscript{\cite{#1}}}
% \providecommand{\supercite}[1]{\textsuperscript{\cite{#1}}}
% \let\citep\parencite
% \let\citet\textcite

%% =========================================================================
%% 第十一部分：超链接与交叉引用
%% =========================================================================

%% 11.1 超链接设置
\hypersetup{%
    pdfstartview=FitH,
    CJKbookmarks=true,
    bookmarksnumbered=true,
    bookmarksopen=true,
    colorlinks=true,
    allcolors=black,
    breaklinks=true
}

\pdfstringdefDisableCommands{%
	\def\cftdotfill{ }%
}

%% 11.2 智能交叉引用设置
\newcommand{\setupcref}[2]{%
    \crefformat{#1}{#2##2##1##3}%
    \crefrangeformat{#1}{#2(##3##1##4)\;\~{}\;(##5##2##6)}%
    \crefmultiformat{#1}{#2(##2##1##3)}{ 和~(##2##1##3)}{，(##2##1##3)}{ 和~(##2##1##3)}%
}
\setupcref{theorem}{\mcm@cap@theorem~}
\setupcref{definition}{\mcm@cap@definition~}

%% =========================================================================
%% 第十二部分：文档结构命令
%% =========================================================================

%% 12.1 标题页生成
\renewcommand{\maketitle}{\par
	\begingroup
      \newpage
      \global\@topnum\z@
      \@maketitle
    \endgroup
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}

\def\@maketitle{%
  \newpage
  \setcounter{page}{1}
  {\centering \zihao{3}\bfseries \@title\par}
  \vskip1ex
}

%% 12.2 摘要环境
\renewenvironment{abstract}{%
\if@twocolumn
		\section*{\abstractname}%
	\else
	 \begin{center}%
	{\zihao{4}\bfseries \abstractname\vspace{-.5em}\vspace{\z@}}%
	 \end{center}%
 	 \quotation
	\fi}
	{\if@twocolumn\else\endquotation\newpage\null\fi}

\renewenvironment{quotation}
	{\list{}{\listparindent 2em%
	 \itemindent \listparindent
	 \rightmargin\z@
	 \leftmargin\z@
	 \parsep \z@ \@plus\p@}%
	\item\relax}
	{\endlist}

%% 12.3 关键词命令
\newcommand\keywords[1]{%
	\renewcommand{\mcm@tokens@keywords}{#1}
	\par
	\vskip1ex
	{\noindent\zihao{-4}\heiti\mcm@cap@keywordsname：}~{\mcm@tokens@keywords}
	}

%% =========================================================================
%% 第十三部分：中文常量定义
%% =========================================================================

%% 13.1 数学定理相关常量
\newcommand*{\mcm@cap@definition}{定义}
\newcommand*{\mcm@cap@theorem}{定理}
\newcommand*{\mcm@cap@lemma}{引理}
\newcommand*{\mcm@cap@corollary}{推论}
\newcommand*{\mcm@cap@assumption}{假设}
\newcommand*{\mcm@cap@conjecture}{猜想}
\newcommand*{\mcm@cap@axiom}{公理}
\newcommand*{\mcm@cap@principle}{定律}
\newcommand*{\mcm@cap@problem}{问题}
\newcommand*{\mcm@cap@example}{例}
\newcommand*{\mcm@cap@proof}{证明}
\newcommand*{\mcm@cap@solution}{解}

%% 13.2 文档结构相关常量
\newcommand*{\mcm@cap@contentsname}{目录}
\newcommand*{\mcm@cap@listfigurename}{插图清单}
\newcommand*{\mcm@cap@listtablename}{附表清单}
\newcommand*{\mcm@cap@refname}{参考文献}
\newcommand*{\mcm@cap@indexname}{索引}
\newcommand*{\mcm@cap@appendixname}{附录}
\newcommand*{\mcm@cap@abstractname}{摘要}
\newcommand*{\mcm@cap@keywordsname}{关键字}

%% 13.3 图表相关常量
\newcommand*{\mcm@cap@figurename}{图}
\newcommand*{\mcm@cap@tablename}{表}

\endinput

%% =========================================================================
%% 文档类结束
%% ========================================================================= 