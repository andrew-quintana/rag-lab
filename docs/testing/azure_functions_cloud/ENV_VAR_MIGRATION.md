# Environment Variable Migration: DATABASE_URL → SUPABASE_DB_URL

**Date**: 2026-01-16  
**Status**: ✅ **COMPLETED** - Code updated and functions redeployed

---

## 📋 Changes Made

### Code Updates

1. **`backend/src/core/config.py`**
   - Updated to use `SUPABASE_DB_URL` with fallback to `DATABASE_URL` for backward compatibility
   - Line 97: `database_url=os.getenv("SUPABASE_DB_URL", os.getenv("DATABASE_URL", ""))`

2. **`backend/azure_functions/backend/src/core/config.py`**
   - Updated to use `SUPABASE_DB_URL` with fallback to `DATABASE_URL`
   - Line 91: `database_url=os.getenv("SUPABASE_DB_URL", os.getenv("DATABASE_URL", ""))`

3. **Test Files Updated**:
   - `backend/tests/conftest.py` - Updated error messages
   - `backend/tests/monitoring/test_function_monitoring.py` - Updated variable name
   - `backend/tests/components/rag/test_rag_generation.py` - Updated error messages
   - `backend/tests/components/rag/test_rag_logging.py` - Updated error messages
   - `backend/scripts/verify_phase5_migrations.py` - Updated error messages

4. **Scripts Updated**:
   - `backend/scripts/set_function_app_env_vars.sh` - Now sets `SUPABASE_DB_URL` with fallback

### Azure Functions Environment Variables

- ✅ **Added**: `SUPABASE_DB_URL` (migrated from `DATABASE_URL`)
- ✅ **Removed**: `DATABASE_URL` (old variable)

### Deployment

- ✅ Built Azure Functions deployment package
- ✅ Deployed to `func-raglab-uploadworkers`
- ✅ Function app restarted

---

## 🔄 Backward Compatibility

The code supports **both** variable names for backward compatibility:
- Primary: `SUPABASE_DB_URL`
- Fallback: `DATABASE_URL` (if `SUPABASE_DB_URL` is not set)

This ensures:
- Existing deployments continue to work
- New deployments use the new variable name
- Gradual migration is possible

---

## ✅ Verification

To verify the migration:

```bash
# Check environment variables
az functionapp config appsettings list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[?name=='SUPABASE_DB_URL' || name=='DATABASE_URL']" -o table

# Should show:
# - SUPABASE_DB_URL: <connection string>
# - DATABASE_URL: (should not exist)
```

---

## 📝 Notes

- **DATABASE_PASSWORD**: Not used in codebase - password is embedded in connection string
- If you want to separate password, you would need to:
  1. Add `SUPABASE_DB_PASSWORD` environment variable
  2. Update connection string construction in code
  3. Parse connection string to replace password

---

## 🎯 Next Steps

1. ✅ Code updated
2. ✅ Azure Functions environment variables migrated
3. ✅ Functions redeployed
4. ⏭️ Test database connection with new variable name
5. ⏭️ Update `.env.prod` and `.env.local` files to use `SUPABASE_DB_URL`

---

**Last Updated**: 2026-01-16
