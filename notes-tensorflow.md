Tensorflow
==========

Tensorflow on alun pitäen Googlen matriisinmurskaukseen kehitetty kirjasto. Tästä luonnollisena seurauksena se päätyi myös koneoppimiskirjastoksi. Erityisenä painotuksena on laskennan hajauttaminen useammalle GPU:lle.

Tensori on n-ulotteinen yleistys matriiseista. Tensorflow:ssa muodosta (shape) puhuttaessa puhutaan listasta josta ilmenee tensorin koko kussakin ulottuvuudessa, ja ulottuvuuksien määrä.

Nykyisen Tensorflow-kirjaston mukana tulee Keras-kirjasto, joka tarjoaa olio-ohjelmointirajapinnan koneoppimiseen.

Keras-kirjastolla perus-neuroverkko on mahdollista määritellä näin:

    model = keras.Sequential([
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(512, activation=tf.nn.relu),
      keras.layers.Dense(10, activation=tf.nn.sigmoid)
    ])

Tässä määrittelyssä input-kerros jää implisiittiseksi. Ensimmäinen piilokerros on mallia "Flatten", joka ottaa moniulotteisen tensorin, tässä tapauksessa 28x28 pikselin kuvan, ja purkaa sen yksiulotteiseksi tensoriksi jossa on 784 elementtiä.

Seuraavat kaksi kerrosta ovat Dense-mallisia, joissa jokainen neuroni ottaa syötteekseen jokaisen edellisen kerroksen neuronin. Näille on määritetty aktivaatiofunkitot, ensimmäisenä Rectifying Linear Unit, joka palauttaa sisääntulonsa jos sisääntulo on positiivinen ja nollan muuten. Toisen kerroksen aktivaatiofunktio on sigmoidi, eli (1+e^(-x))^(-1).

Koulutusparametrit alustetaan kutsumalla neuroverkon compile-funktiota:

    model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='mean_squared_error',
              metrics=['accuracy'])

Optimizer määrittää miten neuroverkon painotuksia muokataan, loss on funktio jonka optimoija pyrkii minimoimaan ja metrics kertoo mitä lisätietoja mallia kouluttaessa ja testatessa mitataan. Jotkin loss-funktiot eivät sulata moniulotteisia tensoreita.

Lopulta neuroverkko koulutetaan komennolla

   model.fit(train\_images, train\_labels, epochs=5)

Jossa ensin annetaan syötteet, seuraavana odotetut ulostulot ja kolmantena montako kierrosta koulutusta jatketaan.
