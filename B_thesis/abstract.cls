%
%  This class file is used for making abstract of Master/graduation thesis.
%    Ver 1.00  2004.1.29
%    
%   by jun ota
%   revised by Takafumi Kanazawa
%

\ProvidesClass{abstract}[2004/01/29]
%
\LoadClassWithOptions{jsarticle}

\def\baselinestretch{0.8}
\setlength{\abovedisplayskip}{0.5mm plus 0.5mm minus 0.5mm}
\setlength{\belowdisplayskip}{0.5mm plus 0.5mm minus 0.5mm}
\setlength{\abovedisplayshortskip}{0.5mm}
\setlength{\belowdisplayshortskip}{0.5mm}

%%% define paper layout
\ifnum\@ptsize=0% for 10pt
\textwidth 178mm%
\textheight 257mm%
\voffset -1in%垂直方向のオフセット
\hoffset -1in%水平方向のオフセット
\topmargin 0mm%オフセットからヘッダまでのマージン
\headheight 18mm%ヘッダの高さ
\headsep 0mm%ヘッダと本文領域の間の高さ
\oddsidemargin 16mm%奇数ページのマージン
\footskip 0mm%文章領域の終わりからフッタの終わりまでの高さ
\fi
\ifnum\@ptsize=-1% for 9pt
\textwidth 194mm% < 180*1.09529
\textheight 280mm% < 257*1.09529
\voffset -1in%垂直方向のオフセット
\hoffset -1in%水平方向のオフセット
\topmargin 0mm%オフセットからヘッダまでのマージン
\headheight 19mm%ヘッダの高さ
\headsep 0mm%ヘッダと本文領域の間の高さ
\oddsidemargin 15mm%奇数ページのマージン
\footskip 0mm%文章領域の終わりからフッタの終わりまでの高さ
\fi

\pagestyle{empty}


\newif\if@abstenglish
\@abstenglishfalse
\newcommand{\englishtitle}{
	\@abstenglishtrue
	\newcommand{\jptitle}[1]{\gdef\@jptitle{##1}}
}

\newcommand{\absttitle}{
	\if@twocolumn
		\twocolumn[\@absttitle]
	\else
		\@absttitle
	\fi
    \global\let\absttitle\relax
    \global\let\@author\@empty
    \global\let\@title\@empty
    \global\let\title\relax
    \global\let\author\relax
    \global\let\and\relax
    \global\let\studentid\@empty%  
    \global\let\lab\@empty
    \global\let\studentid\relax%  
    \global\let\lab\relax

  }
\def\@absttitle{%
    \begin{center}%
			\if@abstenglish
      	{\bf{\LARGE \@title}}\\%
				\underline{\makebox[.925\textwidth][c]{\bf{\LARGE （\@jptitle）}}}\\%
			\else
      	\underline{\bf{\LARGE \@title}}\\%
			\fi
      	{\Large %\bf{
        \begin{tabular}[t]{ccc}%
          学籍番号：\@studentid & \@lab\ 研究室 &\@author 
        \end{tabular}\par}%}%
    \end{center}%
%    \par\vskip -5mm
 \gdef\@studentid{}
 \gdef\@lab{}
%    \ifvoid\@abstractbox\else\centerline{\box\@abstractbox}\vskip1.5em\fi
}
\long\def\studentid#1{\long\gdef\@studentid{#1}}
\long\def\lab#1{\long\gdef\@lab{#1}}

