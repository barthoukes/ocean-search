import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { SearchService, SearchResult } from './search.service';

@Component({
  selector: 'app-root',
  standalone: true,  // This makes it standalone
  imports: [CommonModule, FormsModule, HttpClientModule],  // Import what you need
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
  providers: [SearchService]  // Provide the service here
})
export class AppComponent {
  searchQuery = '';
  results: SearchResult[] = [];
  isLoading = false;
  searched = false;
  searchTime = 0;

  constructor(private searchService: SearchService) {}

  onSearch() {
    if (!this.searchQuery.trim() || this.isLoading) return;
    
    this.isLoading = true;
    this.searched = true;
    const startTime = performance.now();
    
    this.searchService.search(this.searchQuery).subscribe({
      next: (results) => {
        this.results = results;
        const endTime = performance.now();
        this.searchTime = parseFloat(((endTime - startTime) / 1000).toFixed(2));
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Search error:', error);
        this.isLoading = false;
        this.results = [];
      }
    });
  }

  onRandomSearch() {
    const randomQueries = ['document', 'vector', 'search', 'database', 'embedding', 'AI'];
    const randomQuery = randomQueries[Math.floor(Math.random() * randomQueries.length)];
    this.searchQuery = randomQuery;
    this.onSearch();
  }

  openFile(result: SearchResult) {
    window.open(`http://localhost:5000/api/file/${encodeURIComponent(result.filepath)}`, '_blank');
  }
}

