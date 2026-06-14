// GET /api/manifest?id=<participant_id>&stage=<pre|train|post>
//
// Returns the ordered list of stimuli (with assignment ids) for the given
// participant and test_stage. Materialises research_assignments on first call.

import { NextResponse } from "next/server";
import { getOrCreateManifest } from "@/lib/test-assignment";

const STAGE_RE = /^(pre|train|post)$/;
const ID_RE = /^[0-9a-f-]{36}$/i;

export async function GET(req: Request) {
  if (!process.env.SUPABASE_SERVICE_ROLE_KEY) {
    return errorResponse(503, "SUPABASE_SERVICE_ROLE_KEY is not set on the server.");
  }

  const url = new URL(req.url);
  const participantId = (url.searchParams.get("id") ?? "").trim();
  const stage = (url.searchParams.get("stage") ?? "").trim();

  if (!ID_RE.test(participantId)) {
    return errorResponse(400, "Invalid id.");
  }
  if (!STAGE_RE.test(stage)) {
    return errorResponse(400, "stage must be pre, train, or post.");
  }

  try {
    const { manifest, participant } = await getOrCreateManifest(
      participantId,
      stage as "pre" | "train" | "post",
    );
    return NextResponse.json({
      participant: {
        id: participant.id,
        external_code: participant.external_code,
        latin_square_group: participant.latin_square_group,
      },
      stage,
      count: manifest.length,
      manifest,
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (/research_stimuli is empty/.test(msg)) {
      return errorResponse(503, msg);
    }
    if (/participant not found/.test(msg)) {
      return errorResponse(404, msg);
    }
    return errorResponse(500, msg);
  }
}

function errorResponse(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}
