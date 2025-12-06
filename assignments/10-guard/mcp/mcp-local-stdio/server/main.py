#!/usr/bin/env python3
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Annotated, Literal
import pandas as pd
from pydantic import Field

from mcp.server.fastmcp import FastMCP
from sample_data import get_sample_data, get_statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ticket-mcp-server")

# Path to the Excel database
TICKETS_DB_PATH = Path(__file__).parent / "data" / "requests.xls"

class TicketDatabase:
    def __init__(self, excel_path: Path):
        self.excel_path = excel_path
        self._df: Optional[pd.DataFrame] = None
    
    def load_data(self) -> pd.DataFrame:
        """Load ticket data from Excel file."""
        try:
            if not self.excel_path.exists():
                logger.warning(f"Excel file not found at {self.excel_path}")
                return pd.DataFrame()
            
            # Read Excel file
            df = pd.read_excel(self.excel_path)
            logger.info(f"Loaded {len(df)} tickets from database")
            return df
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            return pd.DataFrame()
    
    def search_tickets(self, 
                      user_id: Optional[str] = None,
                      status: Optional[str] = None,
                      priority: Optional[str] = None,
                      category: Optional[str] = None,
                      keyword: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search tickets based on various criteria."""
        df = self.load_data()
        
        if df.empty:
            return []
        
        # Apply filters
        if user_id:
            df = df[df['user_id'].astype(str).str.contains(str(user_id), case=False, na=False)]
        
        if status:
            df = df[df['status'].astype(str).str.contains(str(status), case=False, na=False)]
        
        if priority:
            df = df[df['priority'].astype(str).str.contains(str(priority), case=False, na=False)]
        
        if category:
            # Check if category column exists (for backward compatibility)
            if 'category' in df.columns:
                df = df[df['category'].astype(str).str.contains(str(category), case=False, na=False)]
        
        if keyword:
            # Search in title and description
            keyword_mask = (
                df['title'].astype(str).str.contains(str(keyword), case=False, na=False) |
                df['description'].astype(str).str.contains(str(keyword), case=False, na=False)
            )
            df = df[keyword_mask]
        
        # Convert to list of dictionaries
        return df.to_dict('records')

# Initialize ticket database
ticket_db = TicketDatabase(TICKETS_DB_PATH)

# Create FastMCP server with dependencies
mcp = FastMCP("ticket-mcp-server", dependencies=["pandas>=2.0.0", "openpyxl>=3.1.0"])

@mcp.tool(
    name="search_stickets",
    description="Search and retrieve user support tickets from the database with flexible filtering options",
)
def search_tickets(
    user_id: Annotated[
        str | None, 
        Field(
            description="User ID to search for (partial matches supported)",
            examples=["USR001", "user123"]
        )
    ] = None,
    status: Annotated[
        Literal["open", "closed", "pending", "in_progress"] | None,
        Field(
            description="Filter by ticket status"
        )
    ] = None,
    priority: Annotated[
        Literal["low", "medium", "high", "critical"] | None,
        Field(
            description="Filter by ticket priority level"
        )
    ] = None,
    category: Annotated[
        Literal["authentication", "billing", "feature", "technical", "security"] | None,
        Field(
            description="Filter by ticket category/department"
        )
    ] = None,
    keyword: Annotated[
        str | None,
        Field(
            description="Search keyword in ticket title or description (case-insensitive)",
            min_length=2,
            max_length=100,
            examples=["login issue", "payment failed", "bug report"]
        )
    ] = None
) -> str:
    """Search user tickets in the support database.
    
    This tool allows you to search through customer support tickets using various
    filters. You can combine multiple filters to narrow down results. All text
    searches are case-insensitive and support partial matching.
    
    Args:
        user_id: User ID to search for (supports partial matching)
        status: Ticket status filter
        priority: Ticket priority level filter  
        category: Ticket category/department filter
        keyword: Search term for title and description fields
    
    Returns:
        Formatted string with ticket details, or message if no tickets found
    """
    # Search tickets
    tickets = ticket_db.search_tickets(
        user_id=user_id,
        status=status,
        priority=priority,
        category=category,
        keyword=keyword
    )
    
    if not tickets:
        return "No tickets found matching the search criteria."
    
    # Format results
    result_text = f"Found {len(tickets)} ticket(s):\n\n"
    
    for i, ticket in enumerate(tickets, 1):
        result_text += f"**Ticket #{i}:**\n"
        result_text += f"- ID: {ticket.get('ticket_id', 'N/A')}\n"
        result_text += f"- User ID: {ticket.get('user_id', 'N/A')}\n"
        result_text += f"- Title: {ticket.get('title', 'N/A')}\n"
        result_text += f"- Status: {ticket.get('status', 'N/A')}\n"
        result_text += f"- Priority: {ticket.get('priority', 'N/A')}\n"
        if 'category' in ticket and pd.notna(ticket.get('category')):
            result_text += f"- Category: {ticket.get('category', 'N/A')}\n"
        result_text += f"- Created: {ticket.get('created_date', 'N/A')}\n"
        result_text += f"- Updated: {ticket.get('updated_date', 'N/A')}\n"
        result_text += f"- Description: {ticket.get('description', 'N/A')}\n"
        result_text += f"- Assigned To: {ticket.get('assigned_to', 'N/A')}\n\n"
    
    return result_text

def main():
    """Main function to setup sample data."""
    # Ensure data directory exists
    TICKETS_DB_PATH.parent.mkdir(exist_ok=True)
    
    # Create sample data if Excel file doesn't exist
    if not TICKETS_DB_PATH.exists():
        logger.info("Creating comprehensive sample ticket database...")
       
        try:
            sample_data = get_sample_data()
            stats = get_statistics()
            
            logger.info(f"Generated {stats['total_tickets']} sample tickets with:")
            logger.info(f"  - Categories: {', '.join(stats['by_category'].keys())}")
            logger.info(f"  - Statuses: {', '.join(stats['by_status'].keys())}")
            logger.info(f"  - Priorities: {', '.join(stats['by_priority'].keys())}")
            
        except ImportError:
            logger.warning("sample_data.py not found, using basic sample data")
        
        df = pd.DataFrame(sample_data)
        df.to_excel(TICKETS_DB_PATH, index=False)
        logger.info(f"Sample data created at {TICKETS_DB_PATH}")

if __name__ == "__main__":
    main()
