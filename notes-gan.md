Generative Adversarian Neural network
=====================================

Generative Adversarial Neural network, GAN, saa nimensä siitä että koulutusvaiheessa kaksi neuroverkkoa on vastakkain: Yksi generaattori, joka on se mikä prosessista halutaan ulos, ja toinen diskriminaattori joka koulutetaan erottamaan generoitu data aidosta datasta.

Näistä kahdesta generaattori ottaa syötteenään kohinavektorin ja tuottaa dataa, esimerkiksi kuvia, musiikkia tai tweettejä. Generaattoria koulutettaessa optimoinnin kohteena on diskriminaattorin antama tulos generaattorin tuottamalle datalle. Tarkalleen ottaen tavoitteena on että diskriminaattori toteaisi tuotoksen 100% aidoksi.

Diskriminaattori ottaa syötteenään samanlaista dataa kuin generaattori tuottaa ja palauttaa todennäköisyyden jolla annettu data on aitoa. Diskriminaattoria generoidessa syötteenä annetaan koulutusdataa ja generaattorin tuottamia arvoja. Koulutusdatan odotetuiksi paluuarvo on 100% ja generoidun paluuarvo 0%. Diskriminaattoria siis optimoidaan eri suuntaan kuin generaattoria.

Generaattorista on myös esimerkkejä jotka ottavat syötteenään kohinavektorin lisäksi lisätietoja generoitavasta datasta. On esimerkiksi mahdollista kouluttaa generaattori joka luo kuvan annetun kuvatekstin perusteella. Toinen esimerkki ottaa syötteenään valokuvan ja muuntaa sen Vincent Van Gochin tyyliseksi maalaukseksi.


