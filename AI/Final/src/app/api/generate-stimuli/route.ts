// POST /api/generate-stimuli   { participant_id }
//   → runs C2 + C3 generation for all 18 train-stage assignments
//   → returns final progress (synchronous; takes ~30–60 s for 12 API calls)
//
// GET  /api/generate-stimuli?id=<participant_id>
//   → returns current progress { total, done }
//
// Idempotency: safe to call multiple times; previously-completed
// assignments are skipped server-side.

import { NextResponse } from "next/server";
import {
  generateStimuliForParticipant,
  getGenerateProgress,
} from "@/lib/generate-stimuli";

const ID_RE = /^[0-9a-f-]{36}$/i;

export const maxDuration = 300; // 5 min — Vercel will allow up to this for Pro/Hobby

export async function POST(req: Request) {
  let body: { participant_id?: unknown };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Body must be JSON." }, { status: 400 });
  }
  const participantId =
    typeof body.participant_id === "string" ? body.participant_id.trim() : "";
  if (!ID_RE.test(participantId)) {
    return NextResponse.json({ error: "Invalid participant_id." }, { status: 400 });
  }

  try {
    const result = await generateStimuliForParticipant(participantId);
    return NextResponse.json(result, {
      status: result.failed > 0 && result.done === 0 ? 502 : 200,
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (/calibration not complete|participant not found/i.test(msg)) {
      return NextResponse.json({ error: msg }, { status: 409 });
    }
    if (/Missing API keys/i.test(msg)) {
      return NextResponse.json({ error: msg }, { status: 503 });
    }
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}

export async function GET(req: Request) {
  const url = new URL(req.url);
  const participantId = (url.searchParams.get("id") ?? "").trim();
  if (!ID_RE.test(participantId)) {
    return NextResponse.json({ error: "Invalid id." }, { status: 400 });
  }
  try {
    const progress = await getGenerateProgress(participantId);
    return NextResponse.json(progress);
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : String(err) },
      { status: 500 },
    );
  }
}
