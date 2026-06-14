// 5 calibration segments (Mandarin Chinese, each ~30-50 syllables / ~30-60s read aloud).
//
// Design notes:
//   - Cover all four tones in approximately balanced proportion.
//   - Include common Mandarin phoneme contrasts (zh/z, sh/s, ch/c, n/l, ang/an).
//   - Avoid politically sensitive or dated proper nouns.
//   - Sentences are first-person narrative so the recording feels natural for IVC.
//   - Total target: ~150-250 syllables across 5 segments → enough for stable Fish IVC.

export type CalibrationSegment = {
  id: number;          // 1-indexed
  zh: string;          // Mandarin text to read aloud
  pinyin?: string;     // optional pinyin hint
  topic: string;       // short English label
  syllables: number;   // approximate syllable count
};

export const CALIBRATION_SEGMENTS: CalibrationSegment[] = [
  {
    id: 1,
    zh: "我今天從學校走路回家，沿途看見很多人在公園裡散步、聊天和運動。空氣有點冷，但陽光剛好，讓人覺得很舒服。",
    topic: "Today's walk home",
    syllables: 48,
  },
  {
    id: 2,
    zh: "我最喜歡的一本書是關於一位旅行作家，他花了三年的時間，獨自走遍了許多陌生的城市，並把每一段旅程都仔細地記錄下來。",
    topic: "Favorite book",
    syllables: 52,
  },
  {
    id: 3,
    zh: "週末的時候，我通常會跟朋友一起去看電影或是吃飯，有時候也會留在家裡看書、聽音樂，享受一個安靜的下午。",
    topic: "Weekend routine",
    syllables: 47,
  },
  {
    id: 4,
    zh: "學英文最困難的地方是發音，特別是那些中文沒有的子音，例如 th、v、和短母音，這些聲音常常需要練習很多次才會習慣。",
    topic: "Learning English",
    syllables: 49,
  },
  {
    id: 5,
    zh: "如果未來有機會，我希望可以到不同的國家生活一段時間，認識當地的文化、語言和食物，也順便挑戰自己的舒適圈。",
    topic: "Future hopes",
    syllables: 48,
  },
];

export const CALIBRATION_RECORDING_LIMIT_SEC = 60;
export const CALIBRATION_RECORDING_WARN_SEC = 50;
export const CALIBRATION_RECORDING_MIN_SEC = 8;
