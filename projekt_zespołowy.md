## Projekt zespołowy: wykorzystanie AI w Cyfrowym Przetwarzaniu Obrazów

Wybrać JEDEN konkretny problem/zagadnienie przetwarzania obrazów oraz porównać minimum dwie metody, gdzie conajmniej jedna z nich jest oparta o AI (a w zasadzie Deep Learning/ głębokie sieci neuronowe (DL)).

Co do wyboru problemów/zagadnień metod: przykładowa lista na końcu. Można wybrać coś z poza niej, po wcześniejszym uzgodnieniu.

**Rozmiar grupy: 3-4 osoby**

**Czas na przedstawienie wyników: 20 minut**

**Każda grupa powinna wybrać odrębne metody, problemy mogą się powtarzać.**. Traktujemy różne modele rozwiązujące dany problem (np. różne wersje YOLO do segmentacji) jako odrębne metody.

**Termin przedstawienia:** najpóźniej **2026-06-09**

**Termin przesłania projektu:** przed przedstawieniem wyników.

### Wymagania

Rozwiązanie zadania powinno się sładać z:

- prezentacji

- części technicznej (kod realizujący dane zadanie)


**Nie wystarczy sama prezentacja lub część techniczna.** W pracy trzeba umieć zastosować daną metodę do rozwiązania danego problemu, ale też umieć przedstawić zasadę działania. Jedna grupa uczy się tylko swoich metod.

**Można zawrzeć część prezentacji razem z częścią techniczną, jeśli uda się to płynnie i czytelnie przedstawić.** Np. w formie jednego notatnika.

#### Prezentacja:

- krótko przedstawienie problemu (np. co to jest wykrywanie krawędzi?)

- krótko obszary zastosowań

- krótko opis porównanych metod (zaczynamy od metody "klasycznej" (nie AI)) i porównanie charakterystyk

    - przy wykorzystywaniu dużych, gotowych modeli DL, nie jest wymagane wchodzenie w głębokie szczegóły architektury sieci, ale trzeba opisać, jak działa pipeline (jakiego formatu danych oczekuje model, w jakim formacie produkuje wynik i co trzeba zrobić z danymi przed i po urchomieniu modelu, aby uzyskać finalny wynik)

    - przy porównywaniu dwóch metod DL, trzeba na tyle opisać użyte modele, aby było wiadomo, czym się różnią.

    - opis DL: na czym był trenowany (ogólnie), główne charakterystyki  np. odróżniające go od typowej sieci CNN

- **pamiętać o źródłach.**

#### Część techniczna:

- **Pokazanie wyników**: w przypadku korzystania z gotowych rozwiązań (pretrained models): pokazać działanie metod na 2-3 konretnych przykładach. Fajnie jakby się udało znaleźć coś, co pokazywałoby, po co mamy różne rodzaje metod - gdzie działa X, a nie działa Y, różne poziomy trudności.

   - testowe przykłady wygenerować samemu (zaszumić obraz/zmniejszyć kontrast, w zależności od zadania),

   - przy niektórych przykładach - rekomendowane zbiory z "ground truth" do przeprowadzenia testu podano w linkach. W tym wypadku robimy test minimum na 2-3 przykładach tego typu danych. Dodanie własnego przykładu ponad to (dla którego nie ma punktu odniesienia, co powinno wyjść) z krótką analizą na plus.

   - **porównanie metod:** dotarcie do określonych, obiektywnych liczbowych metryk jakości przystosowanych do danego zadania - wymóg przy większości metod ( np. F1, precision, recall, IntersectionOverUnion, SSIM -- **dana metryka ma sens w zależności od rodzaju problemu! Sugestie są zamieszczone przy tematach, ale proszę zrobić research na ten temat.**). Użyte metryki trzeba krótko wyjaśnić w trakcie przedstawiania wyniku

      - w przypadku wyboru tematu polegającego na samodzielnej konstrukcji i nauki modelu (klasyfikacja.b = pkt 6), trzeba pokazać co najmniej średni wynik na zbiorze testowym, 30 przykładów minimum (ktorego model nie widział przy nauce)

      - generalnie: idealnie chcemy minimum pokazać wizualizację wyniku działania + metryki jakości i krótką interpretację/analizę.

- **Kod źródłowy i obliczenia:** zamieścić cały kod, od początkowej konfiguracji do uzyskania wyniku.

   - przy korzystaniu z gotowego kodu - opisać, czy zadziałało od razu? najważniejsze zmiany: co trzeba było zmienić/spatchować żeby zadziałało? Czy trzeba było uruchomić z GPU?

   - **opisać dokładnie zastosowany "soft"** - jaka biblioteka została użyta lub skąd wzięto model (można dopisać do prezentacji)

   - rekomendacja: wybrać coś, co zadziała na `google colab` - to wymusza w pewnym sensie ograniczenie zakresu. **Dla większości tematów: nie trenujemy wielkich modeli od zera** (bez własnego GPU nie ma sensu)

   - dodatkowe biblioteki (na Colab) można doinstalować bez problemu np. komenda typu `!pip install torchvision`


- **też proszę pamiętać o źródłach.** (skąd model/obrazy)

#### Podział odpowiedzialności

**Ustalacie Państwo jasno zakres pracy, za który dana osoba była odpowiedzialna**. W praktyce udziały pewnie będą się przenikać, ale pytania kontrolne dla danej osoby będą dotyczyć konretnej części, do której jest ona przypisana.


## Przykładowe tematy i materiały:

Podział na dany podproblem i wypisanie przykładowych metod, które można zastosować. Przykłady zostały przetestowane (06.05.2026), wszystko działa, ale czasem po drobnych update'ach.:

1. Odszumianie obrazów. Materiały:

    - (AI) `DnCNN`: https://scico.readthedocs.io/en/stable/examples/denoise_dncnn_universal.html

    - (AI) Restormer: Efficient Transformer for High-Resolution Image Restoration:  https://github.com/swz30/Restormer

    - vs filtry uśredniające/ medianowe / "non-local means" / `bm3D`

    - metryki jakości np. PSNR lub MSE pomiędzy obrazem czystym a odszumionym

2. "Super resolution" (zwiększanie rozdzielczości obrazów):


    - (AI) `ESRGAN` :  https://github.com/xinntao/real-esrgan#-quick-inference

        - (szukamy przykładów w notatniku `.ipynb`)

        - (drobny fix wymagany, ale działa): https://github.com/xinntao/Real-ESRGAN/issues/944


    - (AI) `SRCNN`:

        - (działa, wymagane drobne aktualizacje nazw funkcji i argumentów):  https://github.com/kunal-visoulia/Image-Restoration-using-SRCNN

        - (trzeba ściągnąć repozytorium)

        - (szukamy przykładów w notatniku `.ipynb`)

    - vs. różnego typu interpolacje

    - przykłady powyżej zawierają sensowne metryki jakości i sposoby testowania metod



3. Wykrywanie krawędzi:


    -  (AI) Holistically Nested Edge Detection (HED):

        -  (minimalistyczne, ale działa - ściągamy repozytorium): https://github.com/ashukid/hed-edge-detector

    - (AI) DexiNed:

        - (działa, ściągamy repozytorium i wyciągamy najważniejsze linijki z `dexined.py`: https://huggingface.co/opencv/edge_detection_dexined

    - Sobel/ Canny /Laplace może być jako punkt odniesienia. Można dołożyć filtr dolnoprzepustowy (gaussian albo uśredniający) i porównać wpływ na finalną jakość.

    - dobry zbiór danych do porównania jakości metod  : https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/

        - mamy obraz naturalny + parę referencyjnych wariantów krawędzi jako odniesienie ( nie ma jasnej poprawnej odpowiedzi )

        - na podstawie obrazów powstałych po zastosowaniu filtrów/sieci neuronowej, można np. ustalić threshold jasności na istnienie krywanie krawędzi i porównać zgodność naszych krawędzi ze zbiorem referencyjnym: piksel-po-pikselu, binarnymi miarami jakości klasyfikacji - acc, precision, recall itd.


4. Segmentacja:


    -  (AI) modele YOLO z `ultralytics`: https://docs.ultralytics.com/tasks/segment/ (przykłady w `Predict` pokazują solidny punkt startowy)

        - (tip: maski z wyniku predykcji modelu trzeba przeskalować do rozmiaru oryginalnego obrazu, najlepiej z interpolacją metodą najbliższego siąsiada)

    - klasyczne z `skimage` np. `Watershed`, pipeline z thresholding i morfologią z zajęć (do momentu maski binarnej. raczej bez etykietowania obiektów i wyznaczania `bbox`- to już "Wykrywanie Obietków") :  https://scikit-image.org/docs/dev/auto_examples/index.html#segmentation-of-objects

    - metryki jakości: co najmniej Intersection Over Union (IoU) pomiędzy binarnymi maskami ("ground truth" vs przewidziane przez daną metodę. Maska typu unia po wszystkich wykrytych obszarach z metody, albo tylko tych, których przewidziana etykieta tekstowa pasuje do naszych obietków. Albo chociaż jednego, wybranego obszaru, który najlepiej pasuje wizualnie. To porównujemy z ground truth.)

        - https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/

    - zbiór danych do przykładów (z ground truth): najprościej **Oxford-IIIT Pet Dataset**:

        -  https://www.robots.ox.ac.uk/~vgg/data/pets/

        - najprostszy dostęp przez API `torchvision`: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html

        -  dla każdego obrazu, mamy maskę typu `Trimap` ("ground truth"), porównujemy nasze przewidziane maski np. z częścią "foreground".


5. (a) (AI) Klasyfikacja - analiza zachowania modeli klasyfikacyjnych (na 2-3 przykładach).

    - (AI) wiele modeli opartych na zbiorze danych `ImageNet`

        - modele `YOLO-cls` :  https://docs.ultralytics.com/tasks/classify/#models

        - vision transformer:  https://huggingface.co/google/vit-base-patch16-224

        - EfficientNet: https://huggingface.co/google/efficientnet-b0

    - wybieramy 3 ze swoich ulubionych obrazów, o większej złożoności (więcej niż jeden obiekt, różnorodnego typu i rozmiaru), sprawdzamy wynik klasyfikacji, jej sensowność i jej pewność (confidence) dla top 5 przewidywań z modeli (albo więcej, jak widać coś ciekawego). **Koniecznie: badamy odporność przewyidywań na zakłocenia**:

        - wprowadzamy zaburzenia na różnych poziomach, aż do momentu złamania modelu (uzyskania błędnej klasyfikacji), poprzez modyfikację obrazów wejściowych: dodanie szumu o różnej sile, obniżenie jasności/kontrastu, "pikseloza" (poprzez silne zmniejszenie rodzielczości, lub mocną kompresję stratną)

        - **obrazy muszą być z poza zbiorów treningowych znanym modelom**

5. (b) (AI) Klasyfikacja: trening modelu DL od zera (własna architektura z ISTOTNYMI OGRANICZENIAMI):


    - **przede wszystkim, zbiory danych:** wybieramy tylko przypadki, gdzie **jednemu obrazowi przypisana jest tylko jedna klasa.** Dobre "ground truth" do finalnego testowania i nauki to (wybrać min. 30 obrazów z jednego ze zbiorów do test set):

        - OxfordIIPet (klasyfikacja cat vs dog albo na rasy,  przez API `torchvision`: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html)

        - CIFAR-10 (wybrać 3 klasy np. cat, dog, truck: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)

        - Caltech101 (też jest na  `torchvision`, też lepiej na podzbiorze klas)


    - porównanie kilku architektur CNN

    - lub CNN oraz MLP

    - czy też z klasycznymi klasyfikatorami z `scikit-learn` (rekomendowany `RandomForest` przyjmujący spłaszczony obraz)

    - albo CNN vs CNN + "data augumentation" (losowe rotacje/zoomy obrazów, czy też zmiany jasności - w celu powiększenia danych treningowych)

    - CNN głębszy vs płytszy, kernel size 7x7 vs 3x3, itd...

    - dodanie technik Batch Normalization czy Dropout do bazowego modelu CNN

    - ...możliwości jest dużo

    - **ale ograniczamy się do mniejszych modeli**:

      - rekomendowany rozmiar modelu: do 1-2 mln parametrów

      - trening/nauka - raczej max 20-30 epok lub do 10 minut treningu na CPU na google colab.

    - metryki jakości: mierzone na zbiorze testowym,  typowe dla klasyfikacji: F1, accuracy, recall, confusion matrix



6. Poprawa kontrastu/jasności

    - (AI) Zero-DCE: https://github.com/Li-Chongyi/Zero-DCE (działa, ściągamy repo)

    - (AI) EnlightenGAN: https://github.com/arsenyinfo/EnlightenGAN-inference  (po prostu wg instrukcji)

    - vs CLAHE lub własne pipeline'y wykorzystujące metody poznane na zajęciach

    - metryki: można pokazać histogramy jasności przed i po zastosowaniu metod na ciemnych obrazach, policzyć wariancję jasności., pokazać balans barw przed i po

7. Kompresja


    - (AI) wybrana metoda z frameworku CompressAI, przykład:  https://github.com/InterDigitalInc/CompressAI/blob/master/examples/CompressAI%20Inference%20Demo.ipynb

        - (działa po drobnej korekcie formatu wyjściowego z sieci (rozmiaru) )

    - vs JPEG, JPEG2000, webp

    - metryki jakości: zawarte w przykładzie powyżej, także jak je policzyć. przynajmniej BPP (teoretyczny, oszacowany na wyniku sieci vs faktyczny dla obrazów po kodowaniu w danym formacie); lepiej: dodatkowo PSNR gdzie jako "zaszumioną" wersję bierzemy obraz po kompresji.

8. Wykrywanie obiektów

    - (AI) YOLO https://docs.ultralytics.com/tasks/detect/

    - Pipeline'y z zajęć (oparte o wykrywanie krawędzi lub segmentację, zakończone wyznaczeniem bounding boxa)

    - metryki: co najmniej IoU między bounding boxami: przewidzianym a referencyjnym. Dla uproszczenia - można jak przy segmentacji, policzyć IoU pomiędzy unią wszystkich boxów przewidzianych a unią boxów "ground truth", albo chociaż jednym, najbardziej pasującym do danego "ground truth" boxa. Bardziej ambitnie: dopasowujemy każdemu obiektowi z "ground truth" najbardziej pasujący przewidziany box (według IoU) i pokazujemy wynik zbiorczy i średni dla danego obrazu.

        - https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/

    - zbiór danych (ground truth): chyba najprościej PASCAL VOC + api `torchvision`: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.VOCDetection.html

        - zbiór obrazów + adnotacji w XML, zawierających m.in. współrzędne bounding boxów. `torchivsion` (link powyżej) to chyba najprostszy sposób dostępu do tych danych.
