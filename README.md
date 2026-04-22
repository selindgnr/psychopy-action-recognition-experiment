# PsychoPy Action Recognition Experiment

Bu depo, PsychoPy ile hazirlanmis bir insan eylemi tanima deneyini icerir. Deneyde katilimcilar kisa video klipler izler ve gorulen eylemi `1-5` tuslariyla siniflandirir.

Onerilen GitHub depo adi: `psychopy-action-recognition-experiment`

## Deney Ozeti

- Platform: PsychoPy Builder / PsychoPy Python export
- Ana gorev: Video tabanli action recognition
- Siniflar: `JumpingJack`, `Lunges`, `PullUps`, `PushUps`, `Swing`
- Akis: yonerge -> egitim fazi -> test fazi -> performans ozeti

## Ana Dosyalar

- `experimentpsychopy.psyexp`: Ana PsychoPy Builder dosyasi
- `experimentPrerana.py`: Builder'dan export edilmis ana deney scripti
- `ucf5_training_conditions.csv`: Egitim fazi kosul dosyasi
- `sorted_test_list.csv`: Test fazi icin siralanmis ornek liste
- `videos.xlsx`: Video envanteri

## Yardimci Scriptler

- `master_test_list.py`: Klasorlerden video envanteri uretir ve `videos.xlsx` yazar
- `final_test_list_ordered.py`: Dengeli test listesi uretir ve `sorted_test_list.csv` olusturur
- `concise_csv_extractor.py`: Genis PsychoPy cikti dosyalarini daha kisa analiz formatina cevirir

## Klasor Yapisi

- `training/`: Egitim videolari
- `ucf5/`: Ana test video havuzu
- `extra/`: Alternatif / ek uyaran setleri ve ilgili condition dosyalari
- `pilot/`: Pilot surum ve pilot ciktilari
- `data/`: Calistirma sonrasi olusan deney ciktilari

## GitHub Icin Notlar

Bu depo `.gitignore` ile buyuk video dosyalarini ve uretilen cikti dosyalarini disarida birakacak sekilde ayarlandi. Boylece GitHub'a daha temiz ve yonetilebilir bir proje gider.

Git'te tutulmasi onerilenler:

- `*.psyexp`
- ana Python scriptleri
- condition CSV dosyalari
- analiz / yardimci scriptler
- bu README

Git disinda tutulmasi onerilenler:

- ham video uyaranlari
- `data/` altindaki katilimci ciktilari
- `*_lastrun.py` gibi otomatik uretilen dosyalar
- gecici / yedek calisma klasorleri

Uyaran videolarini da GitHub'da saklamak istiyorsan en iyi secenek normal Git yerine Git LFS kullanmak olur.

## Calistirma

1. PsychoPy'yi kur.
2. Gerekli uyaran videolarinin `training/`, `ucf5/` ve gerekiyorsa `extra/` altinda mevcut oldugundan emin ol.
3. `experimentpsychopy.psyexp` dosyasini Builder ile veya `experimentPrerana.py` dosyasini Python ile calistir.

## Sonraki Adimlar

GitHub'a yuklemeden once sunlari yapmani oneririm:

1. `git status` ile hangi dosyalarin eklenecegini kontrol et.
2. Gerekirse depo adini `psychopy-action-recognition-experiment` olarak sec.
3. Ilk commit'i olustur.
4. GitHub'da bos repo acip remote ekle.
5. `git push -u origin main` ile gonder.

Istersen bir sonraki adimda birlikte ilk commit'i hazirlayip GitHub remote'unu da ekleyebiliriz.
