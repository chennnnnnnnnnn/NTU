// GET /api/survey/clips?id=<participant_id>
//
// Returns the participant's assigned clip manifest (ordered), with answered
// status so the client can resume at the first unanswered clip.

import { NextResponse } from "next/server";
import { getOrCreateSurveyItems } from "@/lib/survey-items";

const ID_RE = /^[0-9a-f-]{36}$/i;

export async function GET(req: Request) {
  if (!process.env.SUPABASE_SERVICE_ROLE_KEY) {
    return errorResponse(503, "SUPABASE_SERVICE_ROLE_KEY is not set on the server.");
  }

  const url = new URL(req.url);
  const participantId = (url.searchParams.get("id") ?? "").trim();
  if (!ID_RE.test(participantId)) {
    return errorResponse(400, "Invalid id.");
  }

  try {
    const items = await getOrCreateSurveyItems(participantId);
    return NextResponse.json({ count: items.length, items });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return errorResponse(500, msg);
  }
}

function errorResponse(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}
