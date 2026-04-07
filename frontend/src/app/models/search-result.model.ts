// This defines what each search result looks like
export interface SearchResult {
  filename: string;
  filepath: string;
  content_preview: string;
  match_type: string;
  embedding_type: string;
  snippets: string[];
}

