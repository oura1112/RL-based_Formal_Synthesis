%%
%%abstract.clsについて
%% 2004.2.4
%% written by o-jun
%% revised by Takafumi Kanazawa

このクラスファイルは，修士論文，卒業論文のA4の概要の
ためのクラスファイルです．
基本的に，jsarticle.clsのオプションを引き継いでいるので
jsarticle.clsを使用します．
jsarticle.clsのバージョンによってうまく余白が取れない
ことがあります．その時は自分で余白を定義するか諦めるかしてください．

●文字サイズは，9pt,10ptに対応してます．

●上下左右の余白を次のように設定してます．
	上下：各2cm
	左右：各1.5cm

●タイトル，学籍番号，研究室名，氏名は，
absttitleコマンドにより出力させることができます．

●英語で書いた場合も概要は日本語で，日本語タイトルも必要です．
英語の場合は日本語の場合に追加して
\englishtitle
コマンドを実行し，さらに日本語タイトルを
\jptitle{日本語の題名}
として定義してください．

absttitleコマンドは，以下を参考してください．

例）
\documentclass[a4j,10pt,twocolumn]{abstract}

%%% \begin{document}の前に，各エントリーを記述する

\title{論文のタイトル}	% 論文のタイトル
\author{氏名} 		% 著者
\studentid{01C3456789} 	% 学籍番号
\lab{潮}		% 研究室名

% 英語なら以下2行を定義
%\englishtitle
%\jptitle{日本語のタイトル}  % 日本語のタイトル

\begin{document}
\absttitle 		% 表題の出力

ここから本文．

\end{document}
