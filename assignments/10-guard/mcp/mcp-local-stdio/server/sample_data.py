#!/usr/bin/env python3
"""
Образцы данных для тикетов службы поддержки

Этот файл содержит реалистичные примеры тикетов для обучения
работе с MCP сервером и системами автоматизации поддержки.
"""

from datetime import datetime, timedelta
import random

def generate_sample_tickets():
    """Генерирует образцы тикетов для демонстрации"""
    
    # Базовые данные для генерации
    users = [
        'user123', 'user456', 'user789', 'user999', 'user555',
        'alice_smith', 'bob_jones', 'carol_wilson', 'david_brown', 'emma_davis',
        'frank_miller', 'grace_taylor', 'henry_white', 'ivy_green', 'jack_black',
        'kate_blue', 'liam_red', 'mia_purple', 'noah_orange', 'olivia_pink'
    ]
    
    agents = [
        'support_agent1', 'support_agent2', 'support_agent3', 'support_agent4',
        'dev_team', 'security_team', 'billing_team', 'tech_lead', 'qa_team'
    ]
    
    # Проблемы по категориям
    login_issues = [
        ("Проблема входа в систему", "Не могу войти с правильными учетными данными"),
        ("Забыт пароль", "Система сброса пароля не работает"),
        ("Двухфакторная аутентификация", "Не получаю SMS с кодом подтверждения"),
        ("Блокировка аккаунта", "Аккаунт заблокирован после нескольких попыток входа"),
        ("SSO проблемы", "Единый вход через Google не работает"),
        ("Сессия истекает", "Постоянно выбрасывает из системы"),
        ("Неверный email", "Система говорит что email не зарегистрирован"),
        ("Капча не работает", "Не могу пройти проверку капчи при входе")
    ]
    
    payment_issues = [
        ("Ошибка обработки платежа", "Платеж отклоняется с ошибкой 500"),
        ("Двойное списание", "Деньги списались дважды за один заказ"),
        ("Возврат средств", "Не могу получить возврат за отмененный заказ"),
        ("Проблема с картой", "Карта не принимается системой"),
        ("Валютные проблемы", "Неправильный курс конвертации валют"),
        ("Проблема с PayPal", "Ошибка при оплате через PayPal"),
        ("Подписка не отменяется", "Не могу отменить ежемесячную подписку"),
        ("Налоги неверные", "Неправильно рассчитываются налоги при оплате")
    ]
    
    feature_requests = [
        ("Темная тема", "Добавьте темную тему для удобства"),
        ("Мобильное приложение", "Нужно приложение для смартфонов"),
        ("API для интеграции", "Требуется REST API для интеграции"),
        ("Экспорт данных", "Возможность экспорта данных в Excel"),
        ("Уведомления", "Push-уведомления о важных событиях"),
        ("Поиск по тегам", "Улучшенный поиск с поддержкой тегов"),
        ("Групповые операции", "Массовые операции с записями"),
        ("Календарная интеграция", "Синхронизация с Google Calendar")
    ]
    
    technical_issues = [
        ("Медленная загрузка", "Страницы загружаются очень медленно"),
        ("Ошибка 404", "Некоторые страницы возвращают 404 ошибку"),
        ("Проблема с файлами", "Не могу загрузить файлы больше 5MB"),
        ("База данных недоступна", "Ошибка подключения к базе данных"),
        ("Проблема с кешем", "Старые данные показываются после обновления"),
        ("Браузерная совместимость", "Не работает в браузере Safari"),
        ("JavaScript ошибки", "Консоль показывает ошибки JS"),
        ("Проблема с CDN", "Статические ресурсы не загружаются")
    ]
    
    security_issues = [
        ("Подозрительная активность", "Заметил необычные попытки входа"),
        ("Утечка данных", "Возможная утечка персональных данных"),
        ("Фишинговые письма", "Получаю подозрительные письма от имени компании"),
        ("Слабый пароль", "Система принимает слишком простые пароли"),
        ("GDPR соответствие", "Вопросы по обработке персональных данных"),
        ("Доступ третьих лиц", "Неавторизованный доступ к моему аккаунту"),
        ("Вирус в загрузках", "Загруженный файл содержит вирус"),
        ("SSL сертификат", "Предупреждения о небезопасном соединении")
    ]
    
    # Объединяем все категории
    all_issues = (
        [(cat, *issue) for issue in login_issues for cat in ['authentication']] +
        [(cat, *issue) for issue in payment_issues for cat in ['billing']] +
        [(cat, *issue) for issue in feature_requests for cat in ['feature']] +
        [(cat, *issue) for issue in technical_issues for cat in ['technical']] +
        [(cat, *issue) for issue in security_issues for cat in ['security']]
    )
    
    # Генерируем тикеты
    tickets = []
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(50):  # Генерируем 50 тикетов
        ticket_id = f"TKT-{i+1:03d}"
        user_id = random.choice(users)
        category, title, description = random.choice(all_issues)
        
        # Логика для статусов и приоритетов
        if category == 'security':
            priority = random.choice(['high', 'critical'])
            status = random.choices(['open', 'pending', 'in_progress'], weights=[0.6, 0.3, 0.1])[0]
        elif category == 'billing':
            priority = random.choices(['high', 'medium'], weights=[0.7, 0.3])[0]
            status = random.choices(['open', 'pending', 'closed'], weights=[0.4, 0.4, 0.2])[0]
        elif category == 'feature':
            priority = random.choices(['low', 'medium'], weights=[0.6, 0.4])[0]
            status = random.choices(['open', 'closed', 'rejected'], weights=[0.3, 0.5, 0.2])[0]
        elif category == 'technical':
            priority = random.choices(['medium', 'high'], weights=[0.6, 0.4])[0]
            status = random.choices(['open', 'in_progress', 'closed'], weights=[0.3, 0.4, 0.3])[0]
        else:  # authentication
            priority = random.choices(['medium', 'high'], weights=[0.5, 0.5])[0]
            status = random.choices(['open', 'pending', 'closed'], weights=[0.4, 0.3, 0.3])[0]
        
        # Генерируем даты
        created_date = base_date + timedelta(days=random.randint(0, 30), 
                                           hours=random.randint(0, 23),
                                           minutes=random.randint(0, 59))
        
        updated_date = created_date + timedelta(hours=random.randint(1, 72))
        
        # Назначение агентов по категориям
        if category == 'security':
            assigned_to = 'security_team'
        elif category == 'billing':
            assigned_to = 'billing_team'
        elif category == 'feature':
            assigned_to = random.choice(['dev_team', 'tech_lead'])
        elif category == 'technical':
            assigned_to = random.choice(['support_agent1', 'support_agent2', 'dev_team'])
        else:
            assigned_to = random.choice(['support_agent1', 'support_agent2', 'support_agent3'])
        
        tickets.append({
            'ticket_id': ticket_id,
            'user_id': user_id,
            'title': title,
            'description': description,
            'status': status,
            'priority': priority,
            'category': category,
            'created_date': created_date.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_date': updated_date.strftime('%Y-%m-%d %H:%M:%S'),
            'assigned_to': assigned_to
        })
    
    return tickets

def get_sample_data():
    """Возвращает структурированные данные для создания DataFrame"""
    tickets = generate_sample_tickets()
    
    # Преобразуем в формат для pandas DataFrame
    data = {
        'ticket_id': [t['ticket_id'] for t in tickets],
        'user_id': [t['user_id'] for t in tickets],
        'title': [t['title'] for t in tickets],
        'description': [t['description'] for t in tickets],
        'status': [t['status'] for t in tickets],
        'priority': [t['priority'] for t in tickets],
        'category': [t['category'] for t in tickets],
        'created_date': [t['created_date'] for t in tickets],
        'updated_date': [t['updated_date'] for t in tickets],
        'assigned_to': [t['assigned_to'] for t in tickets]
    }
    
    return data

def get_statistics():
    """Возвращает статистику по сгенерированным данным"""
    tickets = generate_sample_tickets()
    
    stats = {
        'total_tickets': len(tickets),
        'by_status': {},
        'by_priority': {},
        'by_category': {},
        'by_agent': {}
    }
    
    for ticket in tickets:
        # Статистика по статусам
        status = ticket['status']
        stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
        
        # Статистика по приоритетам
        priority = ticket['priority']
        stats['by_priority'][priority] = stats['by_priority'].get(priority, 0) + 1
        
        # Статистика по категориям
        category = ticket['category']
        stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
        
        # Статистика по агентам
        agent = ticket['assigned_to']
        stats['by_agent'][agent] = stats['by_agent'].get(agent, 0) + 1
    
    return stats

if __name__ == "__main__":
    # Тест генерации данных
    print("🎫 Генерация образцов тикетов...")
    data = get_sample_data()
    print(f"✅ Сгенерировано {len(data['ticket_id'])} тикетов")
    
    print("\n📊 Статистика:")
    stats = get_statistics()
    for category, counts in stats.items():
        if category != 'total_tickets':
            print(f"\n{category.replace('_', ' ').title()}:")
            for item, count in counts.items():
                print(f"  {item}: {count}") 