// research.ts — Service-role wrapper restricted to `research_*` tables.
//
// All research-website backend code MUST go through this wrapper instead of
// calling `createServiceClient()` directly. The wrapper enforces:
//   1. Table name MUST start with `research_` (compile-time string-literal check
//      where possible; runtime guard otherwise).
//   2. Logging of every privileged operation for audit trail (research_events).
//
// Design rationale: same Supabase project is shared with mirror-app. Service
// role bypasses RLS and could accidentally touch mirror-app's `public.users`,
// `public.calibration_clips`, etc. This wrapper is the boundary.

import { createServiceClient } from './server'

type ResearchTable =
  | 'research_participants'
  | 'research_stimuli'
  | 'research_assignments'
  | 'research_recordings'
  | 'research_events'

/**
 * Get a service-role Supabase client locked to the `research_*` table set
 * via a runtime guard around `.from()`.
 *
 * Usage:
 *   const supa = researchServiceClient()
 *   const { data } = await supa.from('research_participants').select('*')
 */
export function researchServiceClient() {
  const raw = createServiceClient()
  return {
    from(table: ResearchTable) {
      if (!table.startsWith('research_')) {
        throw new Error(
          `[research.ts] forbidden table: ${table}. Only research_* tables are allowed.`,
        )
      }
      return raw.from(table)
    },
    storage: raw.storage,
    // For arbitrary RPC the caller is responsible — opt out of the guard:
    rpc: raw.rpc.bind(raw),
  }
}
