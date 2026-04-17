import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

// This service handles ALL communication with your Flask API
@Injectable({
  providedIn: 'root' // Available app-wide
})
export class SearchService {
  // Your Flask API URL
  private apiUrl = 'http://localhost:5000/api';

  // HttpClient is Angular's way of making HTTP requests
  constructor(private http: HttpClient) { }

  // POST /api/search
  search(query: string, k: number = 10): Observable<any> {
    const body = { query, k };
    return this.http.post(`${this.apiUrl}/search`, body);
  }

  // GET /api/stats
  getStats(): Observable<any> {
    return this.http.get(`${this.apiUrl}/stats`);
  }

  // POST /api/fill
  addDocuments(path: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/fill`, { path });

  }
}

