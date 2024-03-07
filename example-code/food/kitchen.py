class Kitchen:
    def __init__(self, ingredients):
        self.ingredients = ingredients

    def sousChef(self):
        """The sous chef magically prepares all ingredients, chopping them finely."""
        self.ingredients = [ingredient + " (chopped)" for ingredient in self.ingredients]
    
    def list_ingredients(self):
        """Lists all ingredients ready for cooking."""
        return ', '.join(self.ingredients)

# Using the Kitchen
my_kitchen = Kitchen(["carrots", "onions", "potatoes"])
print("Original ingredients:", my_kitchen.list_ingredients())

# Letting the sousChef prepare the ingredients
my_kitchen.sousChef()
print("After sousChef's magic:", my_kitchen.list_ingredients())

