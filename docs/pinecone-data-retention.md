# Pinecone Data Retention Issues

## Why Records May Disappear from Pinecone

There are several reasons why ingested records might reduce or disappear from Pinecone over time:

### 1. **Inactive Index Deletion (Free Tier)**
- **Issue**: Pinecone's free Starter plan automatically deletes indexes that are inactive for **14+ days**
- **Solution**: 
  - Regularly query or update your index to keep it active
  - Consider upgrading to a paid plan for persistent storage
  - Monitor index activity and set up alerts

### 2. **Eventual Consistency**
- **Issue**: Pinecone uses eventual consistency - changes may not be immediately visible
- **Solution**: 
  - Wait a few seconds after upsert operations before querying
  - Use `describe_index_stats()` to verify record counts
  - Implement retry logic for critical queries

### 3. **Upsert Overwriting**
- **Issue**: If you upsert vectors with the same `id`, they will **overwrite** existing records
- **Solution**: 
  - Ensure chunk IDs are unique across all documents
  - Use composite IDs like `{doc_id}_{chunk_id}` to prevent collisions
  - Log when overwriting occurs

### 4. **Namespace Issues**
- **Issue**: Data might be in different namespaces, making it appear missing
- **Solution**: 
  - Always specify the correct namespace when querying
  - Check all namespaces using `describe_index_stats()`
  - Use consistent namespace naming

### 5. **Index Recreation**
- **Issue**: If the index is deleted and recreated, all data is lost
- **Solution**: 
  - The code checks if index exists before creating (prevents accidental deletion)
  - Monitor for index deletion events
  - Consider backing up critical data

## Monitoring and Prevention

### Check Index Stats
```python
from src.storage.vector.pinecone import PineconeVectorStore
from src.config.retrieval import PineconeConfig

config = PineconeConfig.from_env()
store = PineconeVectorStore(config)
stats = store.index.describe_index_stats()
print(f"Total vectors: {stats.get('total_vector_count', 0)}")
print(f"Namespaces: {stats.get('namespaces', {})}")
```

### Keep Index Active
- Run periodic queries (even small ones) to keep the index active
- Set up a cron job or scheduled task to query the index every 7-10 days
- Monitor index activity in Pinecone dashboard

### Best Practices
1. **Use unique IDs**: `{doc_id}_{chunk_id}_{timestamp}` or similar
2. **Monitor regularly**: Check index stats after ingestion
3. **Log operations**: Track all upsert/delete operations
4. **Backup critical data**: Store chunks locally (already done in LocalChunkStore)
5. **Handle eventual consistency**: Add delays after upserts before querying

## Current Implementation

The codebase already:
- ✅ Stores chunks locally in `LocalChunkStore` (backup)
- ✅ Logs index stats after initialization
- ✅ Checks if index exists before creating (prevents accidental deletion)
- ✅ Uses consistent namespace ("children" by default)

## Recommendations

1. **Add periodic index stats logging** to track record counts over time
2. **Implement index activity monitoring** to prevent 14-day deletion
3. **Add unique ID generation** to prevent overwrites
4. **Create index health check** utility function
5. **Document namespace usage** clearly in code

