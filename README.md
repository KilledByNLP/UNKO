# UNKO: UNKO is Not Keras Optimization
UNKOは、岩佐 幸翠が作成した深層学習の為の簡易的なライブラリです。
UNKO: UNKO is Not Keras Optimizationという名称は、UNKOがKerasなどと同様に非常に簡潔で可読性の
高いモデル構築が可能であるにも関わらず、Kerasなどと異なり深層学習フレームワークのラッパーではなく
純粋な深層学習ライブラリであることを指し示しています。

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
