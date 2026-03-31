# ocean-search
Search engine based on Python/AI with a vector database.

Todo:
1) **Add meta data**, like the age of the file, the name of the files, directory, writers etc.
2) **Advanced Query Parsing**: Implement more sophisticated query parsing to handle complex natural language queries.
3) **Error Handling**: Add error handling for cases where the query cannot be parsed.
4) **Performance**: Optimize the search process to handle large datasets efficiently.

<img width="1200" height="896" alt="openart-image_1774458048054_958a80e8_1774458048112_2e92ee08" src="https://github.com/user-attachments/assets/1a8a5cda-177d-4c23-a31c-45a53138e468" />

# Start program
```
cd ocean-search/

./ocean_search.py 
⚠️ sentence-transformers not installed. BERT support disabled.
   To enable BERT for text files: pip install sentence-transformers
📝 Smart Embedder: Using Ollama for all file types
🔍 Interactive Document Search with Smart Embedding
============================================================
Commands:
  fill <path>   - add documents from <path> to the database
  clear         - delete ALL documents from the database
  stats         - show database statistics
  <query>       - search for documents matching the query
  q             - quit

Supported extensions: .bash, .bmp, .c, .cc, .cpp, .css, .gif, .go, .h, .hpp, .html, .java, .jpeg, .jpg, .js, .json, .kt, .lua, .md, .pdf, .php, .png, .py, .rb, .rs, .rst, .sh, .text, .tiff, .ts, .txt, .xml, .yaml, .yml

🔧 Using Ollama for all file types

📁 Database exists at: /home/mensfort/workspace/ocean-search/chroma_db
   Use 'clear' to delete all documents and start fresh
   Use 'stats' to see document count

💡 Tip: Results show highlighted matching text to verify relevance
   🟢 Green = exact phrase match, 🟡 Yellow = keyword match
```

# Clear previous data
```
>  clear

⚠️  WARNING: This will delete ALL documents from the database at:
   /home/mensfort/workspace/ocean-search/chroma_db
   Are you sure? (yes/no): yes

🗑️  Clearing database...
   ⚠️  Could not delete collection: At least one of ids, where, or where_document must be provided in delete.
   ✓ Deleted database directory: ./chroma_db
   🔄 Reinitializing database...

✅ Database cleared successfully!
   You can now add new documents with 'fill <path>'
```

# Create a database with files to find
Choose any directory, e.g. /etc
```
> fill /etc 
Error loading markup file /etc/netplan/50-cloud-init.yaml: [Errno 13] Permission denied: '/etc/netplan/50-cloud-init.yaml'
Error loading text file /etc/brltty/Input/vs/all.txt: 'utf-8' codec can't decode byte 0xe9 in position 96: invalid continuation byte
Error loading text file /etc/brltty/Input/lt/all.txt: 'utf-8' codec can't decode byte 0xe5 in position 97: invalid continuation byte

📊 Indexing Summary:
   ✅ Added 94 files to the database
   ⚠️  Skipped 1 empty files (no meaningful content)
   📝 94 files with searchable content
   🏷️  0 files indexed by metadata only
   🧠 18 natural language text files embedded with BERT
   🔧 76 files embedded with Ollama
   ⚠️  Skipped 2870 unsupported files

> 
```

# Now find something in the /etc directory
Lets find the text UTF-9 or similar
```
> UTF-9

📋 Top 5 results for: "UTF-9"
============================================================

1. type-apple.xml
   📂 Path: /etc/ImageMagick-6/type-apple.xml
   🏷️  Type: 📝 MARKUP.xml 🔧
   🔍 Match: ✓ Match from file content
   🔧 Embedded with: Ollama

   🎯 Matching Text Snippets:
      📌 [1] <?xml version="1.0" encoding="UTF-8"?> <!DOCTYPE typemap [   <!ELEMENT typemap (type)+>   <!ATTLIST typemap xmlns CDATA #FIXED ''>   
          Matched: utf

   --------------------------------------------------

2. xfce4-clipman-actions.xml
   📂 Path: /etc/xdg/xfce4/panel/xfce4-clipman-actions.xml
   🏷️  Type: 📝 MARKUP.xml 🔧
   🔍 Match: ✓ Match from file content
   🔧 Embedded with: Ollama

   🎯 Matching Text Snippets:
      📌 [1] <?xml version="1.0" encoding="UTF-8"?> <!DOCTYPE actions [ 	<!ELEMENT actions (action+)> 	<!ELEMENT action (name,regex,commands)> 	<!
          Matched: utf

   --------------------------------------------------

3. type-urw-base35.xml
   📂 Path: /etc/ImageMagick-6/type-urw-base35.xml
   🏷️  Type: 📝 MARKUP.xml 🔧
   🔍 Match: ✓ Match from file content
   🔧 Embedded with: Ollama

   🎯 Matching Text Snippets:
      📌 [1] <?xml version="1.0" encoding="UTF-8"?> <!DOCTYPE typemap [   <!ELEMENT typemap (type)+>   <!ATTLIST typemap xmlns CDATA #FIXED ''>   
          Matched: utf

   --------------------------------------------------

4. all.txt
   📂 Path: /etc/brltty/Input/no/all.txt
   🏷️  Type: 📄 TEXT.txt 🧠
   🔍 Match: ✓ Match from file content
   🧠 Embedded with: BERT (semantic search)

   📝 Content Preview: no braille display ...
   --------------------------------------------------

5. rgb.txt
   📂 Path: /etc/X11/rgb.txt
   🏷️  Type: 📄 TEXT.txt 🧠
   🔍 Match: ✓ Match from file content
   🧠 Embedded with: BERT (semantic search)

   📝 Content Preview: ! $Xorg: rgb.txt,v 1.3 2000/08/17 19:54:00 cpqbld Exp $ 255 250 250		snow 248 248 255		ghost white 248 248 255GhostWhite 245 245 245		white smoke 24...

> 
```

