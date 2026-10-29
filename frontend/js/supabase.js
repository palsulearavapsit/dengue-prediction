import { createClient } from '@supabase/supabase-js';
import { SUPABASE_URL, SUPABASE_ANON_KEY } from './config.js';

// Ensure valid strings to prevent uncaught initialisation exception
const validUrl = (typeof SUPABASE_URL === 'string' && SUPABASE_URL.startsWith('http')) 
  ? SUPABASE_URL 
  : 'https://placeholder.supabase.co';

const validKey = (typeof SUPABASE_ANON_KEY === 'string' && SUPABASE_ANON_KEY.length > 0)
  ? SUPABASE_ANON_KEY
  : 'placeholder-key';

export const isSupabaseConfigured = Boolean(
  typeof SUPABASE_URL === 'string' && SUPABASE_URL.startsWith('http') && 
  typeof SUPABASE_ANON_KEY === 'string' && SUPABASE_ANON_KEY.length > 20
);

export const supabase = createClient(validUrl, validKey);

