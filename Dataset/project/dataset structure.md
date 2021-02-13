## Table of content in datasets

/Data
│   
├── /colourvideo_seg.h5
│     ├── /barefoot
│     │     ├──────── data: shape (320, 300, 180, 320, 3)
│     │     ├──────── metadata: shape (320, 3)
│     │     └──────── timestamps: shape (320, 300)
│     ├── /shod_common
│     │     ├── /fast
│     │     │     ├──── data: shape (290, 300, 180, 320, 3)
│     │     │     ├──── metadata: shape (320, 3)
│     │     │     └──── timestamps: shape (320, 300)
│     │     ├── /natural
│     │     │     ├──── data: shape (290, 300, 180, 320, 3)
│     │     │     ├──── metadata: shape (320, 3)
│     │     │     └──── timestamps: shape (320, 300)
│     │     └── /slow
│     │           ├──── data: shape (290, 300, 180, 320, 3)
│     │           ├──── metadata: shape (320, 3)
│     │           └──── timestamps: shape (320, 300)
│     ├── /shod_other
│     │     ├──────── data: shape (320, 300, 180, 320, 3)
│     │     ├──────── metadata: shape (320, 3)
│     │     └──────── timestamps: shape (320, 300)
│     └── /surveydata: shape (5, 7)
│
│
│
│
│
│
│
├── /colourvideo_sil.h5
│     ├── /barefoot
│     │     ├──────── data: shape (320, 300, 150, 100, 3)
│     │     ├──────── metadata: shape (320, 3)
│     │     └──────── timestamps: shape (320, 300)
│     ├── /shod_common
│     │     ├── /fast
│     │     │     ├──── data: shape (290, 300, 150, 100, 3)
│     │     │     ├──── metadata: shape (290, 3)
│     │     │     └──── timestamps: shape (290, 300)
│     │     ├── /natural
│     │     │     ├──── data: shape (320, 300, 150, 100, 3)
│     │     │     ├──── metadata: shape (320, 3)
│     │     │     └──── timestamps: shape (320, 300)
│     │     └── /slow
│     │           ├──── data: shape (325, 300, 150, 100, 3)
│     │           ├──── metadata: shape (325, 3)
│     │           └──── timestamps: shape (325, 300)
│     ├── /shod_other
│     │     ├──────── data: shape (320, 300, 180, 320, 3)
│     │     ├──────── metadata: shape (320, 3)
│     │     └──────── timestamps: shape (320, 300)
│     └── /surveydata: shape (5, 7)
│
│
│
│
│
├── /depthvideo_seg.h5
├── /depthvideo_sil.h5
│     ├── /barefoot
│     │     ├──────── data: shape (320, 300, 150, 100)
│     │     ├──────── metadata: shape (320, 3)
│     │     └──────── timestamps: shape (320, 300)
│     ├── /shod_common
│     │     ├── /fast
│     │     │     ├──── data: shape (290, 300, 150, 100)
│     │     │     ├──── metadata: shape (290, 3)
│     │     │     └──── timestamps: shape (290, 300)
│     │     ├── /natural
│     │     │     ├──── data: shape (320, 300, 150, 100)
│     │     │     ├──── metadata: shape (320, 3)
│     │     │     └──── timestamps: shape (320, 300)
│     │     └── /slow
│     │           ├──── data: shape (325, 300, 150, 100)
│     │           ├──── metadata: shape (325, 3)
│     │           └──── timestamps: shape (325, 300)
│     ├── /shod_other
│     │     ├──────── data: shape (320, 300, 150, 100)
│     │     ├──────── metadata: shape (320, 3)
│     │     └──────── timestamps: shape (320, 300)
│     └── /surveydata: shape (5, 7)
│
│
│
│
│
├── footpressures_align.h5
│     ├── /barefoot
│     │     ├──────── data: shape (1745, 200, 80, 60)
│     │     ├──────── metadata: shape (1745, 10)
│     │     └──────── timestamps: shape (1745, 200)
│     ├── /shod_common
│     │     ├── /fast
│     │     │     ├──── data: shape (1380, 200, 80, 60)
│     │     │     ├──── metadata: shape (1380, 10)
│     │     │     └──── timestamps: shape (1380, 200)
│     │     ├── /natural
│     │     │     ├──── data: shape (1640, 200, 80, 60)
│     │     │     ├──── metadata: shape (1640, 10)
│     │     │     └──── timestamps: shape (1640, 200)
│     │     └── /slow
│     │           ├──── data: shape (1800, 200, 80, 60)
│     │           ├──── metadata: shape (1800, 10)
│     │           └──── timestamps: shape (1800, 200)
│     ├── /shod_other
│     │     ├──────── data: shape (1625, 200, 80, 60)
│     │     ├──────── metadata: shape (1625, 10)
│     │     └──────── timestamps: shape (1625, 200)
│     └── /surveydata: shape (5, 7)
│
│
│
│
│
├── /footpressures_seg.h5
│     ├── /barefoot
│     │     ├──────── data: shape (1745, 200, 80, 60)
│     │     ├──────── metadata: shape (1745, 9)
│     │     └──────── timestamps: shape (1745, 200)
│     ├── /shod_common
│     │     ├── /fast
│     │     │     ├──── data: shape (1380, 200, 80, 60)
│     │     │     ├──── metadata: shape (1380, 9)
│     │     │     └──── timestamps: shape (1380, 200)
│     │     ├── /natural
│     │     │     ├──── data: shape (1640, 200, 80, 60)
│     │     │     ├──── metadata: shape (1640, 9)
│     │     │     └──── timestamps: shape (1640, 200)
│     │     └── /slow
│     │           ├──── data: shape (1800, 200, 80, 60)
│     │           ├──── metadata: shape (1800, 9)
│     │           └──── timestamps: shape (1800, 200)
│     ├── /shod_other
│     │     ├──────── data: shape (1625, 200, 80, 60)
│     │     ├──────── metadata: shape (1625, 9)
│     │     └──────── timestamps: shape (1625, 200)
│     └── surveydata: shape (5, 7)
│
│
│
│
│
├── jointpositions.h5
│     ├── barefoot
│     │     ├──────── data: shape (80, 1500, 25, 3)
│     │     ├──────── metadata: shape (80, 2)
│     │     └──────── timestamps: shape (80, 1500)
│     ├── shod_common
│     │     ├── fast
│     │     │     ├──── data: shape (80, 1500, 25, 3)
│     │     │     ├──── metadata: shape (80, 2)
│     │     │     └──── timestamps: shape (80, 1500)
│     │     ├── natural
│     │     │     ├──── data: shape (80, 1500, 25, 3)
│     │     │     ├──── metadata: shape (80, 2)
│     │     │     └──── timestamps: shape (80, 1500)
│     │     └── slow
│     │           ├──── data: shape (80, 1500, 25, 3)
│     │           ├──── metadata: shape (80, 2)
│     │           └──── timestamps: shape (80, 1500)
│     ├── shod_other
│     │     ├──────── data: shape (80, 1500, 25, 3)
│     │     ├──────── metadata: shape (80, 2)
│     │     └──────── timestamps: shape (80, 1500)
│     └── surveydata: shape (5, 7)
│
│
│
│
│
└── jointpositions_seg.h5
      ├── barefoot
      │     ├──────── data: shape (320, 300, 25, 3)
      │     ├──────── metadata: shape (320, 3)
      │     └──────── timestamps: shape (320, 300)
      ├── shod_common
      │     ├── fast
      │     │     ├──── data: shape (290, 300, 25, 3)
      │     │     ├──── metadata: shape (290, 3)
      │     │     └──── timestamps: shape (290, 300)
      │     ├── natural
      │     │     ├──── data: shape (320, 300, 25, 3)
      │     │     ├──── metadata: shape (320, 3)
      │     │     └──── timestamps: shape (320, 300)
      │     └── slow
      │           ├──── data: shape (325, 300, 25, 3)
      │           ├──── metadata: shape (325, 3)
      │           └──── timestamps: shape (325, 300)
      ├── shod_other
      │     ├──────── data: shape (320, 300, 25, 3)
      │     ├──────── metadata: shape (320, 3)
      │     └──────── timestamps: shape (320, 300)
      └── surveydata: shape (5, 7)