import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';  // ✅ Add this for ngIf/ngFor
import { FormsModule } from '@angular/forms';    // ✅ Add this for ngModel
import { SearchService } from '../../services/search.service';
import { SearchResult } from '../../models/search-result.model';

@Component({
  selector: 'app-search', // This becomes <app-search></app-search> in HTML
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.css'],
  standalone: true,
  imports: [CommonModule, FormsModule]
})
export class SearchComponent implements OnInit {
  // Component properties
  searchQuery: string = '';
  searchResults: SearchResult[] = [];
  isLoading: boolean = false;
  errorMessage: string = '';
  documentCount: number = 0;
  directoryPath: string = '';
  addStatus: string = '';

  constructor(private searchService: SearchService) { }

  ngOnInit(): void {
    // Called when component loads
    this.loadStats();
  }

  loadStats(): void {
    this.searchService.getStats().subscribe({
      next: (data) => {
        this.documentCount = data.document_count;
      },
      error: (error) => {
        console.error('Error loading stats:', error);
        this.errorMessage = 'Cannot connect to API. Is Flask running?';
      }
    });
  }

  onSearch(): void {
    if (!this.searchQuery.trim()) {
      return;
    }

    this.isLoading = true;
    this.errorMessage = '';
    
    this.searchService.search(this.searchQuery, 10).subscribe({
      next: (results) => {
        this.searchResults = results;
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Search error:', error);
        this.errorMessage = 'Search failed. Check console for details.';
        this.isLoading = false;
      }
    });
  }

  onAddDocuments(): void {
    if (!this.directoryPath) {
      this.addStatus = 'Please enter a directory path';
      return;
    }

    this.addStatus = 'Adding documents...';
    this.searchService.addDocuments(this.directoryPath).subscribe({
      next: (response) => {
        this.addStatus = response.message;
        this.loadStats(); // Refresh document count
        this.directoryPath = '';
        setTimeout(() => {
          this.addStatus = '';
        }, 3000);
      },
      error: (error) => {
        this.addStatus = 'Error: ' + (error.error?.message || 'Failed to add documents');
        console.error('Add documents error:', error);
      }
    });
  }
}

