#!/usr/bin/env python3
"""
Run database migrations.
Usage: python -m src.run_migration
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text

def run_migration():
    """Run the database migration."""
    
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)
    
    # Fix postgres:// to postgresql:// for SQLAlchemy
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    # Read migration file
    migration_file = Path(__file__).parent.parent / 'migrations' / '001_add_language_keywords.sql'
    
    if not migration_file.exists():
        print(f"ERROR: Migration file not found: {migration_file}")
        sys.exit(1)
    
    print(f"Reading migration from: {migration_file}")
    migration_sql = migration_file.read_text()
    
    # Create engine
    print("Connecting to database...")
    engine = create_engine(database_url)
    
    try:
        with engine.connect() as conn:
            # Split SQL into individual statements
            statements = [s.strip() for s in migration_sql.split(';') if s.strip()]
            
            print(f"Executing {len(statements)} SQL statements...")
            
            for i, statement in enumerate(statements, 1):
                if statement.strip():
                    print(f"\nStatement {i}/{len(statements)}:")
                    print(f"  {statement[:80]}...")
                    try:
                        result = conn.execute(text(statement))
                        conn.commit()
                        
                        # If it's a SELECT, show results
                        if statement.strip().upper().startswith('SELECT'):
                            rows = result.fetchall()
                            if rows:
                                print("  Results:")
                                for row in rows:
                                    print(f"    {row}")
                        
                        print(f"  ✓ Success")
                    except Exception as e:
                        print(f"  ✗ Failed: {e}")
                        # Continue with other statements
            
            print("\n✅ Migration completed successfully!")
            
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)
    finally:
        engine.dispose()


if __name__ == "__main__":
    run_migration()
