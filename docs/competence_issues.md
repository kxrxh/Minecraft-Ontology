# Компетентностные вопросы к онтологии

1. Какой может ли кирка `X` быть использована для добычи руды `Y`?
2. Какие инструменты могут быть использованы для добычи руды `X`?
3. Какой рецепт у предмета `X`?
4. Какая прочность у предмета `X`?
5. Какой урон наносит меч `X`?
6. Требуется ли для крафта предмета `X` предмет `Y`?
7. Сколько нужно предмета `X` для крафта предмета `Y`?
8. Какие предеметы входят в состав сета `X`?
9. Какой DPS у меча `X`?
10. Какие предметы могут быть сделаны из предметов `X` и `Y`?
11. На какой высоте можно найти руду `X`?


## Результаты запросов, соответствующие компетентностным вопросам

### 1. Какой может ли кирка `X` быть использована для добычи руды `Y`?

```================================================================================
Query: Can Diamond Pickaxe mine Iron?
================================================================================

can_mine
--------
  Yes   

Total results: 1
```


### 2. What tools can mine Diamond?

```================================================================================
Query: 2. What tools can mine Diamond?
================================================================================

    tool_name    
-----------------
 Diamond Pickaxe 
  Iron Pickaxe   
Netherite Pickaxe

Total results: 3
```


### 3. What is the recipe for Diamond Boots?

```================================================================================
Query: 3. What is the recipe for Diamond Boots?
================================================================================

                                      grid                                      
--------------------------------------------------------------------------------
[[null, null, null], ["Diamond", null, "Diamond"], ["Diamond", null, "Diamond"]]

Total results: 1
```


### 4. What is the durability of Diamond Pickaxe?

```================================================================================
Query: 4. What is the durability of Diamond Pickaxe?
================================================================================

durability
----------
   1561   

Total results: 1
```


### 5. What is the attack damage of Diamond Sword?

```================================================================================
Query: 5. What is the attack damage of Diamond Sword?
================================================================================

attack_damage
-------------
     7.0     

Total results: 1
```

### 6. Is Iron_Ingot required for crafting Diamond Sword?

```================================================================================
Query: 6. Is Iron_Ingot required for crafting Diamond Sword?
================================================================================

is_required
-----------
   false   

Total results: 1
```

### 7. How many Iron Ingot are needed to craft Iron Pickaxe?

```================================================================================
Query: 7. How many Iron Ingot are needed to craft Iron Pickaxe?
================================================================================

count
-----
  3  

Total results: 1
```


### 8. What items are part of Diamond Set?

```================================================================================
Query: 8. What items are part of Diamond Set?
================================================================================

    piece_name    
------------------
  Diamond Boots   
Diamond Chestplate
  Diamond Helmet  
 Diamond Leggings 

Total results: 4
```


### 9. What is the DPS of Diamond Sword?

```================================================================================
Query: 9. What is the DPS of Diamond Sword?
================================================================================

dps 
----
11.2

Total results: 1
```

### 10. What items can be crafted from Iron_Ingot and Stick?
```================================================================================
Query: 10. What items can be crafted from Iron_Ingot and Stick?
================================================================================

 item_name   | Iron_Ingot_count | Stick_count
---------------------------------------------
  Iron Axe   |        3         |      2     
  Iron Hoe   |        2         |      2     
Iron Pickaxe |        3         |      2     
Iron Shovel  |        1         |      2     
 Iron Sword  |        2         |      1     

Total results: 5
```
