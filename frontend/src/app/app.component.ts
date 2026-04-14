import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SearchComponent } from './components/search/search.component';

@Component({
  selector: 'app-root',
  standalone: true,  // This makes it standalone
  imports: [CommonModule, SearchComponent],  // Import what you need
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'OceanSearch';
  showConfig = false

  toggleConfig() {
    this.showConfig = !this.showConfig;
  }
}

