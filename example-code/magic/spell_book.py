class Spellbook:
    def __init__(self, spells):
        self.spells = spells

    def abracadabra(self):
        """A magical spell that reverses the order of spells in the spellbook."""
        self.spells = self.spells[::-1]
    
    def list_spells(self):
        """Lists all spells in the spellbook."""
        return ', '.join(self.spells)

# Using the Spellbook
my_spellbook = Spellbook(["fireball", "lightning bolt", "heal"])
print("Original spellbook:", my_spellbook.list_spells())

# Casting the abracadabra spell to reverse the spellbook
my_spellbook.abracadabra()
print("After abracadabra:", my_spellbook.list_spells())

