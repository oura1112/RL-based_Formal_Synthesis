# RL-based formal sysnthesis of control policies and supervisors.

chapter1:
ω-オートマトンを用いて，与えられた時相論理制御仕様を正の確率で満たす制御方策を，強化学習に基づいて獲得する．
提案法により既存研究でネックとなっていた報酬のスパース性を緩和．
割引率と報酬値の設定により最大確率で制御仕様を満たす方策を獲得させることも可能．

Automaton.py:
用いるω-automatonの定義

AugAutomaton.py:
提案する拡張オートマトンの定義

Grid_World1.py:
環境の定義と各状態遷移で真となる事象を与える関数の定義

product_Grid_World1.py:
環境と拡張オートマトンの合成積の定義

Agent.py:
学習エージェントの定義

main_product_Grid1.py:
実行ファイル

chapter2:
chapter1で提案した方法のスーパバイザ制御への応用．強化学習とベイズ推論に基づいて制御器を学習する．
chapter1と同様に使用を正の確率で満たし，割引率と報酬設計に設定により最大確率で満たすものを設計可能．
制御器が仕様を満たしつつ最もpermissiveなものになるかどうかは未証明．学習の収束性も未証明．

Automaton.py:
用いるω-automatonの定義

AugAutomaton.py:
提案する拡張オートマトンの定義

Grid_World1.py:
環境の定義と各状態遷移で真となる事象を与える関数の定義

product_Grid_World1.py:
環境と拡張オートマトンの合成積の定義

Supervisor_ambiguous.py:
学習スーパバイザ制御器の定義

main_product_Grid1.py:
実行ファイル


main_thesis:
卒業論文．

abstract_oura2.pdf:
卒論発表のアブストラクト

main_thesis.pdf:
卒業論文


