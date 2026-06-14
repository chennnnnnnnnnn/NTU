// POST /api/survey/enroll
// Body: { code?: string }   (code optional — blank = anonymous)
// Behaviour:
//   - If code given: find existing survey_participants by external_code or create one.
//   - If no code: create an anonymous participant with an auto-generated code.
//   - Return { participant_id, next } where next routes to the right stage.

import { NextResponse } from "next/server";
import { surveyServiceClient } from "@/lib/supabase/survey";
import { BATCH_SELECTORS } from "@/lib/survey-config";

const CODE_RE = /^[A-Z0-9]{4,12}$/;

/** Validate the batch selectors (student_id / stimulus_set / person_num)
 *  against the allowed option sets in survey-config. Returns the cleaned
 *  values, or an error string. */
function parseBatch(
  body: Record<string, unknown>,
): { values: Record<string, string> } | { error: string } {
  const values: Record<string, string> = {};
  for (const sel of BATCH_SELECTORS) {
    const raw = body[sel.key];
    if (raw == null || raw === "") continue;
    const v = String(raw);
    if (!sel.options.some((o) => o.value === v)) {
      return { error: `Invalid ${sel.key}: ${v}` };
    }
    values[sel.key] = v;
  }
  return { values };
}

export async function POST(req: Request) {
  if (!process.env.SUPABASE_SERVICE_ROLE_KEY) {
    return errorResponse(503, "SUPABASE_SERVICE_ROLE_KEY is not set on the server.");
  }

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    body = {};
  }

  const rawCode = typeof body.code === "string" ? body.code.toUpperCase().trim() : "";
  if (rawCode.length > 0 && !CODE_RE.test(rawCode)) {
    return errorResponse(400, "Invalid code format. Use 4–12 uppercase letters or digits.");
  }

  const batch = parseBatch(body);
  if ("error" in batch) return errorResponse(400, batch.error);

  const memberName = typeof body.team_member_name === "string" ? body.team_member_name.trim() : "";
  if (memberName.length === 0) return errorResponse(400, "team_member_name is required.");
  if (memberName.length > 60) return errorResponse(400, "team_member_name too long.");
  const fields = { ...batch.values, team_member_name: memberName };

  const supa = surveyServiceClient();

  // With a code: find or create. Without: always create a fresh anon row.
  if (rawCode.length > 0) {
    const { data: existing, error: selErr } = await supa
      .from("survey_participants")
      .select("id, status")
      .eq("external_code", rawCode)
      .maybeSingle();
    if (selErr) return errorResponse(500, `Database error: ${selErr.message}`);

    if (existing?.id) {
      // Refresh selection + name on re-entry.
      await supa.from("survey_participants").update(fields).eq("id", existing.id);
      return NextResponse.json({
        participant_id: existing.id,
        next: routeForStatus(existing.status as string, existing.id as string),
        status: existing.status,
      });
    }
  }

  const code = rawCode.length > 0 ? rawCode : generateAnonCode();
  const { data: inserted, error: insErr } = await supa
    .from("survey_participants")
    .insert({ external_code: code, status: "enrolled", ...fields })
    .select("id, status")
    .single();
  if (insErr || !inserted) {
    return errorResponse(500, `Could not create participant: ${insErr?.message ?? "unknown"}`);
  }

  return NextResponse.json({
    participant_id: inserted.id,
    next: routeForStatus(inserted.status as string, inserted.id as string),
    status: inserted.status,
  });
}

/** enrolled → profile; profile_done/in_progress → rate; completed → done. */
function routeForStatus(status: string, id: string): string {
  switch (status) {
    case "completed":
      return `/clip-survey/done?id=${id}`;
    case "profile_done":
    case "in_progress":
      return `/clip-survey/rate?id=${id}`;
    case "enrolled":
    default:
      return `/clip-survey/profile?id=${id}`;
  }
}

/** Short readable anonymous code, e.g. "ANON7K2Q". */
function generateAnonCode(): string {
  const alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"; // no ambiguous 0/O/1/I/L
  let s = "";
  for (let i = 0; i < 5; i++) {
    s += alphabet[Math.floor(Math.random() * alphabet.length)];
  }
  return `ANON${s}`;
}

function errorResponse(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}
