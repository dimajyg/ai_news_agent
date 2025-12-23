-- Migration: Add language and keywords columns to telegram_groups
-- Date: 2025-12-22
-- Description: Add support for multi-language posts and keyword filtering

-- Add language column (default to English)
ALTER TABLE telegram_groups 
ADD COLUMN IF NOT EXISTS language VARCHAR(10) DEFAULT 'en' NOT NULL;

-- Add keywords column for content filtering
ALTER TABLE telegram_groups 
ADD COLUMN IF NOT EXISTS keywords TEXT;

-- Update existing groups to have default language
UPDATE telegram_groups 
SET language = 'en' 
WHERE language IS NULL OR language = '';

-- Create index for faster language queries
CREATE INDEX IF NOT EXISTS idx_telegram_groups_language 
ON telegram_groups(language);

-- Verify the changes
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_name = 'telegram_groups'
AND column_name IN ('language', 'keywords')
ORDER BY column_name;
