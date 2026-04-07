import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface SearchResult {
  filename: string;
  filepath: string;
  content_preview: string;
  match_type: string;
  embedding_type: string;
  snippets: { snippet: string; matched_terms: string[] }[];
}

@Injectable({ providedIn: 'root' })
export class SearchService {
  private apiUrl = 'http://localhost:5000/api';
  
  constructor(private http: HttpClient) {}
  
  search(query: string, k: number = 10): Observable<SearchResult[]> {
    return this.http.post<SearchResult[]>(`${this.apiUrl}/search`, { query, k });
  }
  
  getStats(): Observable<any> {
    return this.http.get(`${this.apiUrl}/stats`);
  }
}

