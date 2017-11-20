# お断り
本リポジトリは、東京農工大学大学院 情報工学専攻 知能機械デザイン学特論 中間レポートの為に作成されたジョークを兼ねた深層学習フレームワークです。 
全てCPUを用いて処理されているため、学習及び推論は非常に低速です。   
その為、本リポジトリのソースコードを利用することは推奨されません。   

# UNKO: UNKO is Not Keras Optimization
UNKOは、岩佐 幸翠が作成した深層学習の為の簡易的なフレームワークです。
UNKO: UNKO is Not Keras Optimizationという名称は、UNKOがKerasに強い影響を受けたAPI設計であるにも関わらず、
特にKerasのソースコードをforkしたり、拡張したり、何かしらの優位性があったりする訳では無いことを示しています。

# Requirements
Python 3.5以降を想定して記述されています。
Python 2.x系には対応しておりません。

数値計算の為にnumpy、グラフ描画の為にmatplotlibおよびpandasが必要です。
```
pip install -r requirements.txt
```

# Example
2unitsの入力層、50unitsのSigmoid関数によるActivationを伴う隠れ層、1unitの出力層を持つ3層のモデルは、以下のように記述されます。
```
model = Model(learning_rate=0.01)
model.add(Dense(2, 50, activation=Sigmoid()))
model.add(Dense(50, 1))
```
