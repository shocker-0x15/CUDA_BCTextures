# How to create a texture object for Block Compressed Textures

CUDA 11.5以降、ブロック圧縮(BC)テクスチャー用に専用のenumが用意され、これを用いてBCテクスチャー用のCUDA Arrayを作成できるようになりましたが、CUDAサンプルにもドキュメントにも作成したArrayを用いて正しいテクスチャーオブジェクトを作成する方法が示されていなかったので、このリポジトリで示します。

Since version 11.5, CUDA has introduced dedicated enum members for block-compressed (BC) textures. This has enabled the creation of CUDA arrays for BC textures. However, neither the CUDA documentation nor the sample code has shown the proper way to create a texture object using that array. Therefore this repository shows how to do it.

----
2023 [@Shocker_0x15](https://twitter.com/Shocker_0x15)
