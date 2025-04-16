"""
Adaptive Learning Module for J.A.R.V.I.S.
Enables J.A.R.V.I.S. to learn from interactions and improve over time.
"""

import re
import logging
import random
import time
from collections import Counter
import nltk
from datetime import datetime, timedelta

from jarvis.database import models

class AdaptiveLearning:
    """Implements adaptive learning capabilities for J.A.R.V.I.S."""
    
    def __init__(self):
        """Initialize the Adaptive Learning module."""
        self.logger = logging.getLogger(__name__)
        
        # Get default user
        self.user = models.get_or_create_user()
        
        # Initialize user preferences
        self.user_preferences = models.get_user_preferences(self.user.id) or {}
        
        # Internal counters for statistics
        self.commands_processed = 0
        self.successful_commands = 0
        self.learning_rate = 0.1  # How quickly the system adapts
        
        # Pattern recognition
        self.common_patterns = {}
        self.load_patterns_from_database()
        
        self.logger.info("Adaptive Learning module initialized")
        
    def load_patterns_from_database(self):
        """Load learned patterns from the database."""
        try:
            patterns = models.get_learning_patterns(min_count=2)
            
            for pattern in patterns:
                self.common_patterns[pattern.pattern] = {
                    'count': pattern.count,
                    'success_rate': pattern.success_rate / 100.0,  # Convert percentage to float
                    'example': pattern.example_query
                }
            
            self.logger.info(f"Loaded {len(patterns)} patterns from database")
        except Exception as e:
            self.logger.error(f"Error loading patterns from database: {e}")
    
    def learn_from_interaction(self, query, response, was_successful=True):
        """Learn from a user interaction.
        
        Args:
            query: The user's query text
            response: The system's response text
            was_successful: Whether the response was successful
        """
        self.logger.debug(f"Learning from interaction: '{query[:30]}...' -> Success: {was_successful}")
        
        # Record interaction in the database
        try:
            models.record_interaction(
                user_id=self.user.id,
                query=query,
                response=response,
                was_successful=was_successful
            )
        except Exception as e:
            self.logger.error(f"Error recording interaction: {e}")
        
        # Update internal counters
        self.commands_processed += 1
        if was_successful:
            self.successful_commands += 1
        
        # Extract and learn patterns
        self._extract_and_learn_patterns(query, response, was_successful)
        
    def _extract_and_learn_patterns(self, query, response, was_successful):
        """Extract and learn patterns from an interaction.
        
        Args:
            query: The user's query text
            response: The system's response text
            was_successful: Whether the response was successful
        """
        # Clean and normalize the query
        normalized_query = query.lower().strip()
        
        # Extract key patterns (starting words, command structures, etc.)
        patterns = []
        
        # First few words often indicate intent
        words = normalized_query.split()
        if len(words) >= 2:
            patterns.append(" ".join(words[:2]))
        if len(words) >= 3:
            patterns.append(" ".join(words[:3]))
            
        # Look for question patterns
        if normalized_query.startswith("what"):
            patterns.append("what")
        if normalized_query.startswith("how"):
            patterns.append("how")
        if normalized_query.startswith("can you"):
            patterns.append("can you")
        if normalized_query.startswith("tell me"):
            patterns.append("tell me")
            
        # Look for command patterns
        if "search for" in normalized_query:
            patterns.append("search for")
        if "find" in normalized_query:
            patterns.append("find")
        if "open" in normalized_query:
            patterns.append("open")
            
        # Update database with each pattern
        for pattern in patterns:
            try:
                models.update_learning_pattern(
                    pattern=pattern, 
                    success=was_successful, 
                    query=query, 
                    response=response
                )
            except Exception as e:
                self.logger.error(f"Error updating learning pattern: {e}")
    
    def get_learned_command_suggestions(self, prefix, max_suggestions=3):
        """Get command suggestions based on what the system has learned.
        
        Args:
            prefix: The starting text to find suggestions for
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of suggested commands
        """
        suggestions = []
        prefix_lower = prefix.lower()
        
        try:
            # Get patterns from database
            patterns = models.get_learning_patterns(min_count=3, limit=50)
            
            # Find patterns that match the prefix
            matching_patterns = [
                p for p in patterns 
                if p.pattern.startswith(prefix_lower) and p.success_rate > 50
            ]
            
            # Sort by success rate and count
            matching_patterns.sort(
                key=lambda p: (p.success_rate, p.count), 
                reverse=True
            )
            
            # Get top suggestions
            for pattern in matching_patterns[:max_suggestions]:
                if pattern.example_query:
                    suggestions.append({
                        "command": pattern.example_query,
                        "success_rate": pattern.success_rate
                    })
            
        except Exception as e:
            self.logger.error(f"Error getting command suggestions: {e}")
        
        return suggestions
    
    def get_usage_stats(self):
        """Get statistics about system usage and learning.
        
        Returns:
            Dictionary with statistics
        """
        try:
            # Get stats from database
            session = models.get_session()
            
            # Total commands processed
            total_commands = session.query(models.Interaction).count()
            
            # Successful commands
            successful_commands = session.query(models.Interaction).filter_by(was_successful=True).count()
            
            # Success rate
            success_rate = (successful_commands / total_commands) if total_commands > 0 else 0
            
            # Learned patterns
            learned_patterns = session.query(models.LearningPattern).count()
            
            # User feedback stats
            total_feedback = session.query(models.UserFeedback).count()
            positive_feedback = session.query(models.UserFeedback).filter_by(is_positive=True).count()
            user_satisfaction = (positive_feedback / total_feedback) if total_feedback > 0 else 0
            
            session.close()
            
            return {
                "total_commands": total_commands,
                "successful_commands": successful_commands,
                "success_rate": success_rate,
                "learned_patterns": learned_patterns,
                "total_feedback": total_feedback,
                "positive_feedback": positive_feedback,
                "user_satisfaction": user_satisfaction
            }
        except Exception as e:
            self.logger.error(f"Error getting usage stats: {e}")
            return {
                "total_commands": self.commands_processed,
                "successful_commands": self.successful_commands,
                "success_rate": self.successful_commands / self.commands_processed if self.commands_processed > 0 else 0,
                "error": str(e)
            }
    
    def update_user_preference(self, category, name, value):
        """Update a user preference.
        
        Args:
            category: Preference category
            name: Preference name
            value: Preference value
            
        Returns:
            Success flag
        """
        try:
            # Update database
            success = models.update_user_preference(
                user_id=self.user.id,
                category=category,
                name=name,
                value=value
            )
            
            # Update local cache
            if success:
                if category not in self.user_preferences:
                    self.user_preferences[category] = {}
                self.user_preferences[category][name] = value
                
            return success
        except Exception as e:
            self.logger.error(f"Error updating user preference: {e}")
            return False
    
    def record_user_feedback(self, feedback_text, is_positive=True):
        """Record user feedback.
        
        Args:
            feedback_text: Feedback text
            is_positive: Whether the feedback is positive
            
        Returns:
            Success flag
        """
        try:
            return models.record_user_feedback(
                user_id=self.user.id,
                feedback_text=feedback_text,
                is_positive=is_positive
            )
        except Exception as e:
            self.logger.error(f"Error recording user feedback: {e}")
            return False
    
    def generate_personalized_suggestions(self):
        """Generate personalized suggestions based on usage patterns.
        
        Returns:
            List of suggestions
        """
        suggestions = []
        
        try:
            # Get recent successful interactions
            session = models.get_session()
            recent_interactions = session.query(models.Interaction).filter_by(
                user_id=self.user.id,
                was_successful=True
            ).order_by(
                models.Interaction.timestamp.desc()
            ).limit(50).all()
            
            # Get most used patterns
            popular_patterns = session.query(models.LearningPattern).filter(
                models.LearningPattern.success_rate > 80
            ).order_by(
                models.LearningPattern.count.desc()
            ).limit(10).all()
            
            session.close()
            
            # Generate time-based suggestions
            now = datetime.now()
            if 7 <= now.hour < 10:
                suggestions.append({
                    "message": "Good morning! Would you like to check today's weather?",
                    "type": "time-based"
                })
            elif 12 <= now.hour < 14:
                suggestions.append({
                    "message": "It's lunch time. Want me to suggest some nearby restaurants?",
                    "type": "time-based"
                })
            elif 17 <= now.hour < 19:
                suggestions.append({
                    "message": "It's evening. Would you like me to summarize your day?",
                    "type": "time-based"
                })
            
            # Generate suggestions based on popular patterns
            if popular_patterns:
                for pattern in popular_patterns[:2]:
                    if pattern.example_query:
                        suggestions.append({
                            "message": f"Try asking: \"{pattern.example_query}\"",
                            "type": "pattern-based"
                        })
            
            # Generate suggestions based on user's history
            if recent_interactions:
                # Group by category or pattern
                categories = {}
                for interaction in recent_interactions:
                    if interaction.category:
                        categories[interaction.category] = categories.get(interaction.category, 0) + 1
                
                # Find most used categories
                popular_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
                
                if popular_categories:
                    top_category = popular_categories[0][0]
                    if top_category == "weather":
                        suggestions.append({
                            "message": "You seem interested in weather forecasts. Try asking for a weekly forecast.",
                            "type": "history-based"
                        })
                    elif top_category == "news":
                        suggestions.append({
                            "message": "Based on your interests, you might want to check the latest headlines.",
                            "type": "history-based"
                        })
            
            # Limit the number of suggestions
            random.shuffle(suggestions)
            suggestions = suggestions[:3]
            
        except Exception as e:
            self.logger.error(f"Error generating personalized suggestions: {e}")
            suggestions.append({
                "message": "I'm learning more about your preferences to provide better suggestions.",
                "type": "fallback"
            })
        
        return suggestions