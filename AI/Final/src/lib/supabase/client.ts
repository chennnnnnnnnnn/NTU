// Supabase browser-side client.
// Uses the public anon key only — safe to expose to the browser.
// RLS policies on `research_*` tables enforce server-mediated access for writes.

import { createBrowserClient } from '@supabase/ssr'

export function createClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
  )
}
