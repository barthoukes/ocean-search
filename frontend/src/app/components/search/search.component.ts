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
  searchQuery: string = ''; // What the user wants to find, e.g. Bart
  searchResults: SearchResult[] = [];
  isLoading: boolean = false;
  errorMessage: string = '';
  documentCount: number = 0;
  directoryPath: string = '';
  addStatus: string = '';

   // Pagination properties
  allResults: any[] = [];  // Store all results from API
  currentPage: number = 1; // Which page of items to see on a single page.
  pageSize: number = 10;  // Results per page (like Google)
  totalResults: number = 0;
  totalPages: number = 0;
  pageNumbers: number[] = [];
  displayedResults: SearchResult[] = []; 

  Math = Math;

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

  onShow(event?: Event) 
  {
    if (event) {
      event.preventDefault();
      event.stopPropagation();
    }
  
    // Add some visible feedback
    console.log('Show button clicked!');
    
    // Option 1: Show a temporary message
    this.addStatus = 'Show button clicked!';
    setTimeout(() => {
      if (this.addStatus === 'Show button clicked!') {
        this.addStatus = '';
      }
    }, 2000);
    
    // Option 2: Display an alert (for testing)
    // alert('Show button clicked!');
    
    // Option 3: Log current search results count
    console.log(`Current results: ${this.searchResults.length} documents`);
  }

  onSearch() 
  {
    if (this.isLoading) return; // Prevent multiple searches

    this.isLoading = true; // Circle starts to spin on screen
    this.errorMessage = '';
    this.currentPage = 1;  // Reset to first page on new search
    
    this.searchService.search(this.searchQuery, 100).subscribe({  // Changed from 10 to 100 to get more results
      next: (results) => {
        this.allResults = results;
        this.totalResults = results.length;
        this.totalPages = Math.ceil(this.totalResults / this.pageSize);
        this.updatePageNumbers();
        this.updateDisplayedResults();
        this.isLoading = false; // FInished loading, circle stops spinning.
      
        // Optional: Scroll to results (doesn't affect refresh, just UX)
        const resultsElement = document.querySelector('.results-container');
        if (resultsElement) 
        {
          resultsElement.scrollIntoView({ behavior: 'smooth' });
        }
      }
      ,
      error: (error) => {
        console.error('Search error:', error);
        this.errorMessage = 'Search failed. Check console for details.';
        this.isLoading = false;
      }
    });
  }

updateDisplayedResults(): void {
  const startIndex = (this.currentPage - 1) * this.pageSize;
  const endIndex = startIndex + this.pageSize;
  this.searchResults = this.allResults.slice(startIndex, endIndex);
}

goToPage(page: number): void {
  if (page < 1 || page > this.totalPages) return;
  this.currentPage = page;
  this.updateDisplayedResults();
  this.updatePageNumbers();
  
  // Optional: Scroll to top of results
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

updatePageNumbers(): void {
  // Show up to 100 page numbers, with current page in middle when possible
  const maxPagesToShow = 100;
  let startPage: number, endPage: number;
  
  if (this.totalPages <= maxPagesToShow)  
  {
    startPage = 1;
    endPage = this.totalPages;
  } 
  else 
  {
    const maxPagesBeforeCurrent = Math.floor(maxPagesToShow / 2);
    const maxPagesAfterCurrent = Math.ceil(maxPagesToShow / 2) - 1;
    
    if (this.currentPage <= maxPagesBeforeCurrent) 
    {
      startPage = 1;
      endPage = maxPagesToShow;
    } 
    else if (this.currentPage + maxPagesAfterCurrent >= this.totalPages) 
    {
      startPage = this.totalPages - maxPagesToShow + 1;
      endPage = this.totalPages;
    }
    else
    {
      startPage = this.currentPage - maxPagesBeforeCurrent;
      endPage = this.currentPage + maxPagesAfterCurrent;
    }
  }
  
  this.pageNumbers = Array.from(
    { length: endPage - startPage + 1 }, 
    (_, i) => startPage + i
  );
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

