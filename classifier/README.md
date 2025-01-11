# osu-classifier

osu-classifier is a model that predicts which osu! ranked mapper mapped a given beatmap.

Try the model [here](https://colab.research.google.com/github/OliBomby/Mapperatorinator/blob/main/colab/mapperatorinator_inference.ipynb).

## Usage

Run `classify.py` with the path to the beatmap you want to classify and the time in seconds of the segment you want to use to classify the beatmap.
```shell
python classify.py beatmap_path="'...\Songs\1790119 THE ORAL CIGARETTES - ReI\THE ORAL CIGARETTES - ReI (Sotarks) [Cataclysm.].osu'" time=60
```

```
Mapper: Sotarks (4452992) with confidence: 9.760356903076172
Mapper: Sajinn (13513687) with confidence: 6.975161075592041
Mapper: kowari (5404892) with confidence: 6.800069332122803
Mapper: Haruto (3772301) with confidence: 6.077754020690918
Mapper: Kalibe (3376777) with confidence: 5.894346237182617
Mapper: iljaaz (8501291) with confidence: 5.873990535736084
Mapper: tomadoi (5712451) with confidence: 5.817874431610107
Mapper: Nao Tomori (5364763) with confidence: 5.144880294799805
Mapper: Kujinn (3723568) with confidence: 5.082106590270996
...
```