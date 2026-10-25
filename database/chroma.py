import chromadb

class DBClient:

    def __init__(self, tenant, database, api_key):
        self.tenant =  tenant
        self.database = database
        self.api_key = api_key

    def get_client(self):
        client = chromadb.CloudClient(
            tenant=self.tenant,
            database=self.database,
            api_key=self.api_key,
        )
        return client
